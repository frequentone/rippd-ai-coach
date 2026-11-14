import { supabase } from "../lib/supabase";
import { cache } from "../lib/cache";
import { post } from "../lib/api";
import { getWeeklyStreakSnapshot } from "./streakService";

// ─── Post-workout coach debrief ───────────────────────────────────────────────
// Fired right after a session ends. The backend generates a coach-initiated
// message (overload nudge / skipped check-in / PR praise) directly into the
// chat history, so it appears as the coach texting first. Fire-and-forget.

export type DebriefSet = { weight: number | null; reps: number | null };
export type DebriefExercise = { name: string; sets: DebriefSet[] };

export function sendWorkoutDebrief(payload: {
  user_id: string;
  workout_title: string;
  duration_mins: number;
  exercises: DebriefExercise[];   // completed sets only
  skipped_exercises: string[];    // exercises with zero completed sets
  done_sets: number;
  total_sets: number;
  new_prs: string[];
}): Promise<{ message: string | null }> {
  return post("/workout-debrief", payload);
}

// ─── Local date helper ────────────────────────────────────────────────────────

function localDateStr(date: Date = new Date()): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  return `${y}-${m}-${d}`;
}

// ─── Types ────────────────────────────────────────────────────────────────────

type SliceKey = "nutrition" | "training" | "progress";

export type LiveUserContext = {
  personalBests: Array<{ exercise_name: string; weight_kg: number; reps: number; logged_at: string }>;
  recentWorkouts: Array<{ date: string; exercises_done: number; workoutTitle: string }>;
  avgDailyCalories: number;
  avgDailyProtein: number;
  todayCalories: number;
  todayProtein: number;
  waterGlasses: number;
  targetCalories: number;
  targetProtein: number;
  plannedTomorrow: Array<{ meal_type: string; food_name: string; calories: number }>;
  currentWeight: number | null;
  weightTrend: "gaining" | "losing" | "stable";
  currentStreak: number;
  longestStreak: number;
};

// ─── Module-level cache ───────────────────────────────────────────────────────

const CACHE_TTL_MS = 24 * 60 * 60 * 1000;

const _cache: {
  data: LiveUserContext | null;
  fetchedAt: number | null;
  userId: string | null;
  invalidatedSlices: Set<SliceKey>;
} = {
  data: null,
  fetchedAt: null,
  userId: null,
  invalidatedSlices: new Set(),
};

function isStale(): boolean {
  if (!_cache.fetchedAt) return true;
  return Date.now() - _cache.fetchedAt > CACHE_TTL_MS;
}

// ─── Slice fetchers ───────────────────────────────────────────────────────────

async function fetchNutritionSlice(
  userId: string
): Promise<Pick<LiveUserContext,
  | "avgDailyCalories" | "avgDailyProtein"
  | "todayCalories" | "todayProtein"
  | "waterGlasses" | "targetCalories" | "targetProtein" | "plannedTomorrow">> {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - 7);
  const today = localDateStr();
  const tomorrowDate = new Date();
  tomorrowDate.setDate(tomorrowDate.getDate() + 1);
  const tomorrow = localDateStr(tomorrowDate);

  const [logsRes, waterRes, macrosRes, plannedRes] = await Promise.all([
    supabase
      .from("food_logs")
      .select("calories, protein_g, logged_at")
      .eq("user_id", userId)
      .gte("logged_at", localDateStr(cutoff)),
    supabase
      .from("water_logs")
      .select("glasses")
      .eq("user_id", userId)
      .eq("log_date", today)
      .maybeSingle(),
    supabase
      .from("user_macros")
      .select("calories, protein_g")
      .eq("user_id", userId)
      .maybeSingle(),
    supabase
      .from("planned_meals")
      .select("meal_type, food_name, calories")
      .eq("user_id", userId)
      .eq("plan_date", tomorrow),
  ]);

  const rows = (logsRes.data ?? []) as Array<{ calories: number; protein_g: number; logged_at: string }>;
  const byDate: Record<string, { cal: number; prot: number }> = {};
  for (const row of rows) {
    const key = row.logged_at.slice(0, 10);
    if (!byDate[key]) byDate[key] = { cal: 0, prot: 0 };
    byDate[key].cal += row.calories ?? 0;
    byDate[key].prot += row.protein_g ?? 0;
  }
  const days = Object.values(byDate);
  const todayTotals = byDate[today] ?? { cal: 0, prot: 0 };

  return {
    avgDailyCalories: days.length > 0 ? days.reduce((s, d) => s + d.cal, 0) / days.length : 0,
    avgDailyProtein:  days.length > 0 ? days.reduce((s, d) => s + d.prot, 0) / days.length : 0,
    todayCalories: todayTotals.cal,
    todayProtein: todayTotals.prot,
    waterGlasses: waterRes.data?.glasses ?? 0,
    targetCalories: macrosRes.data?.calories ?? 0,
    targetProtein: macrosRes.data?.protein_g ?? 0,
    plannedTomorrow: (plannedRes.data ?? []) as LiveUserContext["plannedTomorrow"],
  };
}

async function fetchTrainingSlice(
  userId: string
): Promise<Pick<LiveUserContext, "personalBests" | "recentWorkouts">> {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - 7);

  const [pbRes, logsRes] = await Promise.all([
    supabase
      .from("personal_bests")
      .select("exercise_name, weight_kg, reps, logged_at")
      .eq("user_id", userId)
      .order("logged_at", { ascending: false })
      .limit(5),

    supabase
      .from("workout_logs")
      .select("date, exercises_done, workout:workouts(title)")
      .eq("user_id", userId)
      .eq("completed", true)
      .gte("date", localDateStr(cutoff))
      .order("date", { ascending: false }),
  ]);

  return {
    personalBests: (pbRes.data ?? []) as LiveUserContext["personalBests"],
    recentWorkouts: (logsRes.data ?? []).map((row: any) => ({
      date: row.date,
      exercises_done: row.exercises_done ?? 0,
      workoutTitle: row.workout?.title ?? "Workout",
    })),
  };
}

async function fetchProgressSlice(
  userId: string
): Promise<Pick<LiveUserContext, "currentWeight" | "weightTrend" | "currentStreak" | "longestStreak">> {
  const cutoff = new Date();
  cutoff.setDate(cutoff.getDate() - 30);

  const [progressRes, streakSnapshot] = await Promise.all([
    supabase
      .from("progress_logs")
      .select("body_weight, logged_at")
      .eq("user_id", userId)
      .gte("logged_at", localDateStr(cutoff))
      .order("logged_at", { ascending: true }),

    getWeeklyStreakSnapshot(userId),
  ]);

  const logs = (progressRes.data ?? []) as Array<{ body_weight: number; logged_at: string }>;
  let currentWeight: number | null = null;
  let weightTrend: "gaining" | "losing" | "stable" = "stable";
  if (logs.length >= 2) {
    const oldest = logs[0].body_weight;
    const newest = logs[logs.length - 1].body_weight;
    currentWeight = newest;
    const diff = newest - oldest;
    weightTrend = diff > 0.5 ? "gaining" : diff < -0.5 ? "losing" : "stable";
  } else if (logs.length === 1) {
    currentWeight = logs[0].body_weight;
  }

  return {
    currentWeight,
    weightTrend,
    currentStreak: streakSnapshot.currentStreak,
    longestStreak: streakSnapshot.longestStreak,
  };
}

// ─── Main context fetcher (cached) ────────────────────────────────────────────

export async function getUserContext(userId: string): Promise<LiveUserContext> {
  // Different user, full invalidation
  if (_cache.userId !== userId) {
    _cache.data = null;
    _cache.fetchedAt = null;
    _cache.userId = userId;
    _cache.invalidatedSlices = new Set();
  }

  // Full fetch when stale or cold
  if (isStale() || !_cache.data) {
    const [nutrition, training, progress] = await Promise.all([
      fetchNutritionSlice(userId),
      fetchTrainingSlice(userId),
      fetchProgressSlice(userId),
    ]);
    _cache.data = { ...nutrition, ...training, ...progress };
    _cache.fetchedAt = Date.now();
    _cache.invalidatedSlices.clear();
    return _cache.data;
  }

  // Partial refresh for invalidated slices only
  const stale = [..._cache.invalidatedSlices];
  if (stale.length > 0) {
    const patches = await Promise.all(
      stale.map((slice) => {
        if (slice === "nutrition") return fetchNutritionSlice(userId);
        if (slice === "training") return fetchTrainingSlice(userId);
        return fetchProgressSlice(userId);
      })
    );
    for (const patch of patches) Object.assign(_cache.data, patch);
    _cache.invalidatedSlices.clear();
  }

  return _cache.data;
}

// Back-compat alias used by coach.tsx
export const fetchLiveUserContext = getUserContext;

// ─── Cache invalidation ───────────────────────────────────────────────────────

// Call after food is logged, workout is completed, or weight is recorded
// so the next coach request re-fetches only the stale slice.
export function invalidateContextSlice(slice: SliceKey): void {
  _cache.invalidatedSlices.add(slice);
}

// ─── Topic detection (UI only) ────────────────────────────────────────────────
// Used purely to decide which follow-up card to render next to a reply.
//
// It does NOT influence the coach: buildTopicContext / buildUserContextString /
// buildSystemPrompt used to live here and were dead, ai/main.py's ChatRequest
// documents user_context as "intentionally ignored, backend builds it", and the
// persona prompt is assembled server-side. They burned queries every message to
// build strings nothing read, and made coach behaviour look tunable from here.
// It isn't: topic routing lives in ai/context_builder.py (detect_topics) and the
// prompt in ai/rag_pipeline.py. Change coach behaviour there.

export type Topic = "nutrition" | "training" | "progress" | "general";

export function detectTopic(message: string): Topic {
  const lower = message.toLowerCase();

  const nutritionKw = ["food", "eat", "diet", "calories", "protein", "meal", "hungry",
    "recipe", "macro", "carb", "fat", "nutrition", "drink", "calorie", "snack", "cook", "cheat"];
  const trainingKw  = ["workout", "exercise", "lift", "gym", "sets", "reps", "push", "pull",
    "training", "bench", "squat", "deadlift", "run", "cardio", "muscle", "strength", "session"];
  const progressKw  = ["weight", "streak", " pr ", "personal best", "progress", "results",
    "gain", "lose", "body", "measurement", "bmi", "fat loss", "bulk"];

  const score = (kws: string[]) => kws.filter((k) => lower.includes(k)).length;

  const ns = score(nutritionKw);
  const ts = score(trainingKw);
  const ps = score(progressKw);

  const max = Math.max(ns, ts, ps);
  if (max === 0) return "general";
  if (ns === max) return "nutrition";
  if (ts === max) return "training";
  return "progress";
}

// ─── Get followed creators for coach selector ─────────────────────────────────

export async function getFollowedCoaches(userId: string, force = false) {
  const key = `coaches:${userId}`;
  if (!force) {
    const hit = cache.get<any>(key);
    if (hit) return hit;
  }
  const { data, error } = await supabase
    .from("creator_subscriptions")
    .select(
      `
      creator_id,
      creators!creator_subscriptions_creator_id_fkey (
        id,
        name,
        specialty,
        profile_image,
        tone,
        coaching_philosophy,
        diet_philosophy,
        bio
      )
    `
    )
    .eq("user_id", userId)
    .eq("active", true);

  if (error) throw error;
  const result = (data ?? []).map((r: any) => r.creators).filter(Boolean);

  // Creators don't subscribe to themselves, but they should still get their
  // own AI coach persona, same experience their subscribers have.
  if (!result.some((c: any) => c.id === userId)) {
    const { data: self } = await supabase
      .from("creators")
      .select("id, name, specialty, profile_image, tone, coaching_philosophy, diet_philosophy, bio")
      .eq("id", userId)
      .maybeSingle();
    if (self) result.unshift(self);
  }

  cache.set(key, result);
  return result;
}

// ─── Get conversation history for a user+creator pair ────────────────────────

// meta carries whatever card this message rendered, so the thread can redraw
// it on reload. See supabase/ai_conversations_meta.sql.
export type ConversationMeta =
  | { card: "program_proposal"; args: any }
  | { card: "plan_brief"; brief: any }
  | { card: string; [k: string]: any };

export type ConversationRow = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  meta?: ConversationMeta | null;
};

export const CHAT_HISTORY_DAYS = 7;
export const CHAT_PAGE_SIZE = 20;

// Loads ONE page of conversation, scoped to the last 7 days. We query
// newest-first (so a cursor can page backwards through history) then return the
// page oldest → newest for display. Pass `before` (an ISO created_at) to fetch
// the next older page when the user scrolls up. This keeps only a small window
// of messages in memory instead of the entire history.
export async function getConversationPage(
  userId: string,
  creatorId: string,
  before?: string | null,
  limit: number = CHAT_PAGE_SIZE
): Promise<ConversationRow[]> {
  let q = supabase
    .from("ai_conversations")
    .select("id, role, content, created_at, meta")
    .eq("user_id", userId)
    .eq("creator_id", creatorId)
    .in("role", ["user", "assistant"])
    .order("created_at", { ascending: false })
    .limit(limit);

  if (before) q = q.lt("created_at", before);

  const { data, error } = await q;
  if (!error) return ((data ?? []) as ConversationRow[]).reverse();

  // `meta` is a newer column (supabase/ai_conversations_meta.sql). If this app
  // build reaches a database that predates it, PostgREST 400s on the whole
  // select and the user sees an empty thread. Losing the cards is fine; losing
  // the conversation is not — so fall back to the original column list.
  let retry = supabase
    .from("ai_conversations")
    .select("id, role, content, created_at")
    .eq("user_id", userId)
    .eq("creator_id", creatorId)
    .in("role", ["user", "assistant"])
    .order("created_at", { ascending: false })
    .limit(limit);
  if (before) retry = retry.lt("created_at", before);

  const { data: rows, error: retryError } = await retry;
  if (retryError) throw retryError;
  return ((rows ?? []) as ConversationRow[]).reverse();
}

// ─── Pending action (server-side proposal awaiting yes/no) ────────────────────

// The backend stores an unconfirmed tool call as a role='pending' row whose
// content is JSON: {"creator_id", "actions": [{"tool", "args", "intent"}]}.
// Fetching it on load lets the UI re-show the proposal card after a timeout,
// refresh, or app restart instead of degrading to the plain-text fallback.
export type PendingProposal = {
  tool: string;                       // "propose_workout_program" | "propose_diet_plan"
  args: any;                          // matches ProgramProposalData for workout programs
  intent?: string;
};

export async function getPendingProposal(
  userId: string,
  creatorId: string
): Promise<PendingProposal | null> {
  const { data, error } = await supabase
    .from("ai_conversations")
    .select("content")
    .eq("user_id", userId)
    .eq("creator_id", creatorId)
    .eq("role", "pending")
    .order("created_at", { ascending: false })
    .limit(1);

  if (error || !data?.length) return null;
  try {
    const parsed = JSON.parse(data[0].content);
    const action = parsed?.actions?.[0];
    if (action && typeof action === "object" && "tool" in action) {
      return { tool: action.tool, args: action.args, intent: action.intent };
    }
  } catch {
    /* malformed row, ignore */
  }
  return null;
}

// Unconfirmed food logs use the same role='pending' row, but with
// {"type": "log_food", ...macros} actions. Rehydrating them on load lets the
// editable confirmation table survive an app restart instead of silently
// degrading to the "say yes" text path. Only rows fresher than maxAgeHours are
// returned, a stale "confirm your eggs" card from yesterday is worse than none.
export type PendingFoodAction = {
  food_name: string;
  meal_type?: string | null;
  count?: number | null;
  unit?: string | null;
  quantity_g?: number | null;
  calories?: number | null;
  protein_g?: number | null;
  carbs_g?: number | null;
  fat_g?: number | null;
};

export async function getPendingFoodLog(
  userId: string,
  creatorId: string,
  maxAgeHours = 6
): Promise<PendingFoodAction[] | null> {
  const { data, error } = await supabase
    .from("ai_conversations")
    .select("content, created_at")
    .eq("user_id", userId)
    .eq("creator_id", creatorId)
    .eq("role", "pending")
    .order("created_at", { ascending: false })
    .limit(1);

  if (error || !data?.length) return null;
  const ageMs = Date.now() - new Date(data[0].created_at).getTime();
  if (ageMs > maxAgeHours * 3600_000) return null;
  try {
    const parsed = JSON.parse(data[0].content);
    const foods = (parsed?.actions ?? []).filter(
      (a: any) => a && typeof a === "object" && a.type === "log_food"
    );
    return foods.length ? (foods as PendingFoodAction[]) : null;
  } catch {
    /* malformed row, ignore */
  }
  return null;
}

// Deletes the pending row after the user confirms or cancels on the client -
// without this the rehydrated card would reappear on every conversation load.
export async function clearPendingActions(
  userId: string,
  creatorId: string
): Promise<void> {
  await supabase
    .from("ai_conversations")
    .delete()
    .eq("user_id", userId)
    .eq("creator_id", creatorId)
    .eq("role", "pending");
}

export async function updatePendingProposal(
  userId: string,
  creatorId: string,
  proposal: PendingProposal
): Promise<void> {
  const content = JSON.stringify({
    creator_id: creatorId,
    actions: [{ tool: proposal.tool, args: proposal.args, intent: proposal.intent }],
  });
  const { data } = await supabase
    .from("ai_conversations")
    .select("id")
    .eq("user_id", userId)
    .eq("creator_id", creatorId)
    .eq("role", "pending")
    .order("created_at", { ascending: false })
    .limit(1);
  if (data?.length) {
    await supabase.from("ai_conversations").update({ content }).eq("id", data[0].id);
  }
}

export async function clearPendingProposal(
  userId: string,
  creatorId: string
): Promise<void> {
  await supabase
    .from("ai_conversations")
    .delete()
    .eq("user_id", userId)
    .eq("creator_id", creatorId)
    .eq("role", "pending");
}

// ─── Save a message ───────────────────────────────────────────────────────────

export async function saveMessage(
  userId: string,
  creatorId: string,
  role: "user" | "assistant",
  content: string
) {
  const { error } = await supabase.from("ai_conversations").insert({
    user_id: userId,
    creator_id: creatorId,
    role,
    content,
  });
  if (error) throw error;
}
