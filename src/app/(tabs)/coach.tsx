import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Gesture, GestureDetector } from "react-native-gesture-handler";
import { tabSwipeRef } from "../../lib/tabSwipeRef";
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  ScrollView,
  TextInput,
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
} from "react-native";
import { Image } from "expo-image";
import { mediaSource, listThumbFor } from "../../lib/bunny";
import { HAIRLINE } from "../../components/Glass";
import ExerciseDetailModal from "../../components/ExerciseDetailModal";
import { Ionicons } from "@expo/vector-icons";
import { SafeAreaView } from "react-native-safe-area-context";
import { router, useFocusEffect, useLocalSearchParams } from "expo-router";
import { useAuth } from "../../context/AuthContext";
import { useCreator } from "../../context/CreatorContext";
import { logMeal, deleteFoodLog } from "../../services/mealService";
import { replacePlannedMeal } from "../../services/plannedMealService";
import { getWaterGlasses, setWaterGlasses } from "../../services/waterService";
import { invalidateFoodTodayCache } from "../../services/homeService";
import {
  SuggestedMealCards,
  WorkoutSuggestionCard,
  TodayProgressCard,
  WeeklyStatsCard,
  CoachMemoryCard,
  type SuggestedMeal,
  type SuggestedExercise,
  type AddExercisesResult,
  type CandidateFact,
} from "../../components/CoachCards";
import { saveCoachMemoryFacts, archiveCoachMemoryFacts } from "../../services/coachMemoryService";
import BodyMap, { MUSCLE_LABELS } from "../../components/BodyMap";
import { supabase } from "../../lib/supabase";
import { inferMealType, type MealType } from "../../lib/mealType";
import { logBodyWeight } from "../../services/progressService";
import {
  logWorkoutSets,
  getActiveProgramWeek,
  getMyPlan,
  addExerciseToWorkout,
  saveProgramFromProposal,
  swapExerciseInActivePlan,
  type CustomSet,
} from "../../services/workoutPlanService";
import { addTodaySwap } from "../../lib/sessionSwaps";
import { grantXP, XP } from "../../services/xpService";

import {
  getFollowedCoaches,
  getConversationPage,
  getPendingProposal,
  getPendingFoodLog,
  clearPendingActions,
  updatePendingProposal,
  clearPendingProposal,
  saveMessage,
  fetchLiveUserContext,
  detectTopic,
  invalidateContextSlice,
  LiveUserContext,
  CHAT_PAGE_SIZE,
} from "../../services/coachService";
import { getPreferences, getProfile, addToAllergies } from "../../services/profileService";
import { getCreators, followCreator } from "../../services/creatorService";
import { trackAction } from "../../services/analyticsService";
import ConfirmDialog from "../../components/ConfirmDialog";
import { toast } from "../../lib/toast";
import { Exercise, getExercisesByIds, getExerciseLibraryLite, getExerciseAlternatives, getEligibilityCtx } from "../../services/exerciseLibraryService";
import ExerciseDemoPreview from "../../components/ExerciseDemoPreview";
import { pickDemoVideo } from "../../lib/bunny";
import { COLORS } from "../../constants/colors";
import { FONTS } from "../../constants/fonts";
import DragSheet from "../../components/DragSheet";
import FadeSlideIn from "../../components/FadeSlideIn";
import PressableScale from "../../components/PressableScale";
import Skeleton from "../../components/Skeleton";
import { haptic } from "../../lib/haptics";
import Animated, {
  FadeInUp,
  Easing,
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withSequence,
  withTiming,
  withDelay,
} from "react-native-reanimated";
import { get, post } from "../../lib/api";
import { KEYBOARD_DISMISS, useKeyboardVisible } from "../../lib/keyboard";
import { wantsProgramBuild } from "../../lib/coachIntents";
import { storage } from "../../lib/storage";
import { useTabBarHeight } from "../../lib/tabBar";
import {
  ExpoSpeechRecognitionModule,
  useSpeechRecognitionEvent,
} from "expo-speech-recognition";

const COACH_LOADING_MSGS = [
  "Reading your data...",
  "Consulting your coach...",
  "Analyzing your sets...",
  "Checking your progress...",
  "Crunching your numbers...",
  "Building your response...",
  "Reviewing your history...",
  "Almost there...",
];

// Ordered by real daily frequency: do-it-now actions first (eat / train / log),
// then check-ins, then lower-frequency planning. Only tasks needing multiple
// structured inputs the coach can't infer get a wizard (program build, swap) -
// everything else is a one-tap prompt that returns an instant grounded answer.
// Ordered CHEAPEST -> MOST EXPENSIVE in LLM cost. The first chips are the ones
// users tap most (and tap repeatedly), so the free//cheap actions sit up front
// and the heavy generations are a deliberate scroll away — you build a program
// once, you don't want to fire an 8k-token generation by fat-fingering chip #1.
const QUICK_PROMPTS = [
  // Zero tokens — handled entirely on-device.
  "Log my weight",        // local pending-action card, no LLM
  "Cheat meal today",     // 3-question wizard, then one grounded plan
  "Plan my day",          // interactive daily schedule prompt
  "Swap an exercise",     // wizard + exercises_library lookup, no LLM
  // Cheap: tiny context, 2-3 line answers.
  "Feeling sore",
  "How much protein?",
  "I skipped today",
  "How's my recovery?",
  // Medium: pulls progress / nutrition context.
  "Am I on track?",
  "What should I eat?",
  "What should I train today?",
  // Heavy: multi-slice context or full generation. Rare by nature, so being
  // last costs nothing in usability.
  "Review my week",       // a whole week of nutrition + training + progress
  "Adjust my plan",       // wizard, then rewrites the program
  "Build me a program",   // full multi-day generation (~8k tokens, 90s timeout)
];

// Prompts that need a wizard get action: "wizard_X". Others map to a specific
// message that tells the AI to answer directly without asking more questions.
const PROMPT_OVERRIDES: Record<string, string> = {
  "Plan my day":
    "Let's plan my day ahead! Break down a structured schedule for today's meals and workout based on my targets, and ask if I have any special events like eating out.",
  "What should I eat?":
    "What should I eat right now to hit the rest of today's macros? Give me 2-3 specific meal or snack options with rough calories and protein, based on what I've already logged today. No questions.",
  "What should I train today?":
    "What's my workout for today? Pull it from my current program, list the exercises with sets and reps, and add 1-2 quick form cues for the main lift. No questions.",
  "Review my week":
    "Review my week: how did my calories, protein and training go vs my targets, and what's the ONE thing to fix next week? Be direct, 3-4 lines, no questions.",
  "How's my recovery?":
    "Based on my recent training load and rest days, how's my recovery looking? Give me a direct 2-3 line assessment, no questions.",
  "Am I on track?":
    "Am I on track with my fitness goals? Give me a direct yes or no with 2-3 specific data points from my activity. No follow-up questions.",
  "How much protein?":
    "How much protein should I eat daily given my current goal and weight? Give me a specific gram target and a quick reason.",
  "Feeling sore":
    "I'm feeling muscle soreness today. Should I train or rest? Give me a direct recommendation.",
  "I skipped today":
    "I just skipped today's workout. What's the best way to adjust, should I make it up or just move on? Give me a direct answer.",
};

// ─── Types ────────────────────────────────────────────────────────────────────

type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  created_at?: string;
  exercises?: Exercise[];
  programProposal?: ProgramProposalData; // frozen card shown after confirm/cancel
  animate?: boolean; // typewriter reveal for freshly-received replies
  meals?: SuggestedMeal[];         // coach meal suggestions → Log it / Plan it cards
  workoutExercises?: SuggestedExercise[]; // coach exercise recommendations → Add to plan card
  progressCard?: boolean;          // inline "today so far" macro card
  weeklyCard?: boolean;            // inline weekly stats card
  candidateFacts?: CandidateFact[]; // "worth remembering?" coach-memory card
  planBrief?: PlanBriefSummary;    // frozen recap of the muscle-map brief
};

// Stable unique ids for locally-created messages (so React keys stay stable
// across prepends, required for smooth scroll-up pagination).
let _localSeq = 0;
const newLocalId = () => `local-${++_localSeq}`;

type FoodConfirmation = {
  food_name: string;
  count?: number | null;
  unit?: string | null;
  quantity?: number | null;
  calories?: number | null;
  protein_g?: number | null;
  carbs_g?: number | null;
  fats_g?: number | null;
  meal_type?: string | null;
};

type SwapConfirmation = {
  original_exercise?: string;
  new_exercise?: string;
};

type WorkoutSetInput = { weight_kg: number | null; reps: number };
type WorkoutLogExercise = { exercise: string; sets: WorkoutSetInput[] };
type WorkoutLogConfirmation = { exercises: WorkoutLogExercise[] };

type ProgramProposalData = {
  title: string;
  difficulty: string;
  goal: string;
  duration_weeks: number;
  coaching_rationale: string;
  days: Array<{
    day_number: number;
    title: string;
    is_rest_day: boolean;
    exercises?: Array<{ name: string; sets: number; reps: string; rest_seconds?: number }>;
  }>;
  isModify?: boolean;
};

type CorrectionItem = {
  id: string;
  food_name: string;
  quantity_g?: number | null;
  calories?: number | null;
  meal_type?: string | null;
};
type CorrectionConfirmation = {
  mode: "remove" | "correct";
  items: CorrectionItem[];
  corrected?: {
    food_name: string;
    quantity_g?: number | null;
    calories?: number | null;
    protein_g?: number | null;
    carbs_g?: number | null;
    fat_g?: number | null;
    meal_type?: string | null;
  } | null;
};

type PendingAction =
  | { type: "food_log"; data: FoodConfirmation[] }
  | { type: "exercise_swap"; data: SwapConfirmation }
  | { type: "diet_preference"; data: { food_item: string } }
  | { type: "weight_log"; data: { weight_kg: number } }
  | { type: "workout_log"; data: WorkoutLogConfirmation }
  | { type: "program_proposal"; data: ProgramProposalData }
  | { type: "water_log"; data: { glasses: number } }
  | { type: "correction"; data: CorrectionConfirmation };

type ChatResponse = {
  response: string;
  messages_remaining?: number;
  upgrade_required?: boolean;
  upgrade_message?: string;
  intent?: string;
  pending_food_data?: Array<{
    food_name: string;
    count?: number | null;
    unit?: string | null;
    quantity_g?: number | null;
    calories?: number | null;
    protein_g?: number | null;
    carbs_g?: number | null;
    fat_g?: number | null;
    meal_type?: string | null;
  }>;
  pending_swap_data?: {
    original_exercise?: string;
    replacement_exercise?: string;
  };
  recommended_exercises?: Array<{
    exercise_id: string;
    display_name: string;
    slug: string;
  }>;
  pending_weight_data?: { weight_kg: number };
  pending_workout_data?: WorkoutLogConfirmation;
  pending_correction_data?: CorrectionConfirmation;
  pending_tool?: string;
  pending_args?: Omit<ProgramProposalData, "isModify">;
  suggested_meals?: SuggestedMeal[] | null;
  suggested_exercises?: SuggestedExercise[] | null;
  candidate_facts?: CandidateFact[] | null;
};

// ─── Helpers ──────────────────────────────────────────────────────────────────

// Keeps **bold**, drops everything else.
//
// This used to strip bold too, which defeated MdText below, the renderer has
// always supported bold, and every reply arrived with the markers already
// removed, so key numbers ("**180g protein**", "**50kg**") landed as flat prose.
// Headings and code fences stay stripped: a chat bubble is not a document, and
// an <h2> inside one looks broken rather than structured.
function tidyMarkdown(text: string): string {
  return text
    .replace(/__(.*?)__/g, "**$1**")                  // __bold__ → **bold**
    // Italic only when the asterisks are single, a naive /\*(.*?)\*/ matches
    // the empty span inside "**bold**" and eats the bold markers with it.
    .replace(/(?<!\*)\*(?!\*)([^*\n]+)\*(?!\*)/g, "$1")
    .replace(/(?<![a-zA-Z0-9])_(?!_)([^_\n]+)_(?![a-zA-Z0-9])/g, "$1")
    .replace(/`([^`]+)`/g, "$1")                      // inline code
    .replace(/^#{1,6}\s+/gm, "")                      // headings
    .replace(/^\s*[-*+]\s+/gm, "• ")                  // bullets → •
    .trim();
}

// ─── Chat history grouping (day separators + time labels) ─────────────────────

const nowIso = () => new Date().toISOString();

// Local YYYY-MM-DD key for grouping messages by calendar day.
function dayKey(iso?: string): string {
  const d = iso ? new Date(iso) : new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
    d.getDate()
  ).padStart(2, "0")}`;
}

// "Today" / "Yesterday" / "Mon, 16 Jun" for the day separator chips.
function dayLabel(iso?: string): string {
  const d = iso ? new Date(iso) : new Date();
  const today = dayKey();
  const y = new Date();
  y.setDate(y.getDate() - 1);
  if (dayKey(iso) === today) return "Today";
  if (dayKey(iso) === dayKey(y.toISOString())) return "Yesterday";
  return d.toLocaleDateString([], { weekday: "short", day: "numeric", month: "short" });
}

function timeLabel(iso?: string): string {
  if (!iso) return "";
  return new Date(iso).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
}

type ChatItem =
  | { kind: "day"; label: string; key: string }
  | { kind: "msg"; msg: Message; key: string };

// Flattens messages into a render list, inserting a day separator before the
// first message of each calendar day. Always ends on a "Today" separator so a
// fresh day shows an empty "Today" section with the full history scrollable above.
function buildChatItems(messages: Message[]): ChatItem[] {
  const items: ChatItem[] = [];
  let lastDay = "";
  messages.forEach((msg, i) => {
    const k = dayKey(msg.created_at);
    if (k !== lastDay) {
      items.push({ kind: "day", label: dayLabel(msg.created_at), key: `day-${k}` });
      lastDay = k;
    }
    items.push({ kind: "msg", msg, key: `m-${msg.id}` });
  });
  if (lastDay !== dayKey()) {
    items.push({ kind: "day", label: "Today", key: "day-today" });
  }
  return items;
}

// ─── Exercise mention detection ───────────────────────────────────────────────

// Scans an assistant reply for exercise names that appear in the library index
// (sorted longest-name-first so "incline bench press" wins over "bench press").
// Returns up to 3 slugs, skipping overlapping matches so one phrase yields one
// card. Used as a fallback when the backend doesn't attach recommended_exercises.
function detectExerciseSlugs(
  text: string,
  index: { name: string; slug: string }[]
): string[] {
  const lower = text.toLowerCase();
  const slugs: string[] = [];
  const ranges: Array<[number, number]> = [];
  for (const { name, slug } of index) {
    const n = name.toLowerCase();
    // Skip trivially short single words to avoid matching common English words.
    if (n.length < 6 && !n.includes(" ")) continue;
    const idx = lower.indexOf(n);
    if (idx === -1) continue;
    const start = idx;
    const end = idx + n.length;
    // Require word boundaries around the match.
    const before = start > 0 ? lower[start - 1] : " ";
    const after = end < lower.length ? lower[end] : " ";
    if (/[a-z0-9]/.test(before) || /[a-z0-9]/.test(after)) continue;
    // Skip if this match overlaps a longer one already taken.
    if (ranges.some(([s, e]) => start < e && end > s)) continue;
    if (slugs.includes(slug)) continue;
    ranges.push([start, end]);
    slugs.push(slug);
    if (slugs.length >= 3) break;
  }
  return slugs;
}

// ─── Weight log detection ─────────────────────────────────────────────────────
// When the AI responds with "Logged weight: 80.3kg\n\nSay yes to log this or
// no to cancel." we intercept it, strip the confirmation text, and show a card.
// Matches a bare affirmative reply ("yes", "add it", "log them all"…) used to
// confirm a pending meal/exercise suggestion without another model round trip.
const AFFIRM_ADD_RE =
  /^(yes|yeah|yep|yup|sure|ok(ay)?|do it|add (it|them|these|that|all)?( to my (plan|workout|log|diet))?|log (it|them|all)|plan it)[.!]?$/i;

// Matches a bare "forget that" reply right after a coach-memory fact was
// confirmed, archives it locally without a round trip to the backend.
const FORGET_RE = /^(forget (that|it|about that)|remove that|delete that|un-?remember that)[.!]?$/i;

// Detects "I drank 3 glasses of water" / "had 500ml water" / "drank 1L water"
// in the USER's message. Returns glasses (250ml each), capped at 20.
function extractWaterGlasses(text: string): number | null {
  const lower = text.toLowerCase();
  if (!lower.includes("water")) return null;
  if (!/\b(drank|drink|had|log|finished|chugged)\b/.test(lower)) return null;

  const glassM = lower.match(/(\d+(?:\.\d+)?)\s*glass(?:es)?/);
  if (glassM) return Math.min(20, Math.max(1, Math.round(parseFloat(glassM[1]))));

  const mlM = lower.match(/(\d+(?:\.\d+)?)\s*ml/);
  if (mlM) return Math.min(20, Math.max(1, Math.round(parseFloat(mlM[1]) / 250)));

  const lM = lower.match(/(\d+(?:\.\d+)?)\s*(?:l|litres?|liters?)\b/);
  if (lM) return Math.min(20, Math.max(1, Math.round(parseFloat(lM[1]) * 4)));

  if (/\ba glass of water\b/.test(lower)) return 1;
  return null;
}

function extractWeightLog(text: string): { weight_kg: number; cleanText: string } | null {
  const m = text.match(/logged\s+weight[:\s]+(\d+(?:\.\d+)?)\s*kg/i);
  if (!m) return null;
  const weight = parseFloat(m[1]);
  if (!weight || weight < 20 || weight > 500) return null;
  const cleanText = text
    .replace(/\n*say yes to log this or no to cancel\.?/i, "")
    .replace(/\n*say (?:yes|"?yes"?).*cancel\.?/i, "")
    .trim();
  return { weight_kg: weight, cleanText };
}

// ─── Food dislike detection ───────────────────────────────────────────────────

// This runs BEFORE the message is sent to the coach and returns early on a
// match, so anything it claims is a food dislike never reaches the classifier
// at all. That makes a false positive expensive twice over: garbage is written
// into user_preferences.allergies_or_dislikes as a food the coach must never
// serve again, AND the user's real request goes unanswered.
//
// It fired on "remove the barbell bench press from my plan" — `remove` is not a
// food word and `(.+)` swallowed the rest of the sentence — so a user asking to
// change their PROGRAM got a "Never eat this again?" card, and the string "the
// barbell bench press from my plan" ended up in their food restrictions.
// "remove the rice I logged" is the same shape, and that phrasing is literally
// the `correct_log` example in the backend classifier prompt.
//
// So: bail out on anything that smells like training or like a log entry, keep
// only triggers that are actually about eating, and bound the capture — a food
// name is a few words, not a clause.
const NOT_A_FOOD_RE =
  /\b(plan|program|programme|routine|split|workout|exercise|training|session|warm.?up|set|sets|rep|reps|kg|logged|log|entry|diary|bench|squat|deadlift|press|curl|row|pull|push|lunge|plank|cardio|treadmill|dumbbell|barbell|machine|day|leg|legs|chest|back|arm|arms|shoulder|shoulders|abs|glutes|bicep|biceps|tricep|triceps|core)\b/i;

// Articles/quantifiers to strip so "the paneer" and "paneer" store identically.
const FOOD_LEAD_RE = /^(the|any|all|my|some|more)\s+/i;

function detectFoodDislike(message: string): string | null {
  const lower = message.toLowerCase().trim();
  // Training / logging vocabulary anywhere in the sentence disqualifies it —
  // the backend classifier handles those intents properly.
  if (NOT_A_FOOD_RE.test(lower)) return null;
  const patterns = [
    // Explicitly about eating — safe on their own.
    /(?:i don['']t want to eat|i don['']t eat|i can['']t eat|i hate eating|i['']m allergic to|allergic to)\s+(.+)/i,
    // Ambiguous verbs that only survive because NOT_A_FOOD_RE ran above.
    /(?:don['']t suggest|never suggest|stop suggesting|stop adding|never add)\s+(.+)/i,
    /(?:no more|cut out)\s+(.+)/i,
  ];
  for (const pattern of patterns) {
    const match = lower.match(pattern);
    if (match && match[1]) {
      const food = match[1]
        .replace(/[.,!?]+$/, "")
        .replace(FOOD_LEAD_RE, "")
        .trim();
      // A food is a short noun phrase. A clause ("the bench press from my
      // plan") is the failure mode this whole guard exists for.
      if (food.length > 0 && food.length <= 30 && food.split(/\s+/).length <= 4) {
        return food;
      }
    }
  }
  return null;
}

// ─── Confirmation Cards ───────────────────────────────────────────────────────

// One editable macro tile in the 2×2 grid.
function MacroCell({
  label,
  value,
  unit,
  accent,
  onChange,
  onTouch,
}: {
  label: string;
  value: string;
  unit: string;
  accent?: boolean;
  onChange: (v: string) => void;
  onTouch: () => void;
}) {
  return (
    <View style={cs.macroCell}>
      <Text style={cs.macroLbl}>{label}</Text>
      <View style={cs.macroValRow}>
        <TextInput
          style={[cs.macroInput, accent && { color: COLORS.primary }]}
          value={value}
          onChangeText={(t) => {
            onChange(t.replace(/[^0-9.]/g, ""));
            onTouch();
          }}
          keyboardType="decimal-pad"
          placeholder="0"
          placeholderTextColor="#3A3A3A"
          selectTextOnFocus
        />
        <Text style={[cs.macroUnit, accent && { color: COLORS.primary }]}>{unit}</Text>
      </View>
    </View>
  );
}

const MEAL_TYPE_LABEL: Record<string, string> = {
  breakfast: "Breakfast",
  morning_snack: "Morning Snack",
  lunch: "Lunch",
  evening_snack: "Evening Snack",
  dinner: "Dinner",
};

function FoodLogCard({
  data,
  onConfirm,
  onCancel,
}: {
  data: FoodConfirmation[];
  onConfirm: (items: FoodConfirmation[]) => void;
  onCancel: () => void;
}) {
  const [saving, setSaving] = useState(false);
  // baseItems holds the original per-unit macros so scaling stays accurate —
  // it must capture data ONCE and not track later changes.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  const baseItems = useMemo(() => data, []);
  const [items, setItems] = useState<FoodConfirmation[]>(() => data);
  const [touched, setTouched] = useState(false);

  const time = useMemo(
    () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
    []
  );

  const mealLabel = items[0]?.meal_type ? MEAL_TYPE_LABEL[items[0].meal_type] ?? "Meal" : "Meal";

  const updateItem = (i: number, field: keyof FoodConfirmation, raw: string) => {
    const num = parseFloat(raw);
    const val = isNaN(num) ? null : num;
    setItems((prev) => prev.map((item, idx) => idx === i ? { ...item, [field]: val } : item));
    setTouched(true);
  };

  // When count changes, scale all macros proportionally from the original values
  const updateCount = (i: number, raw: string) => {
    const newCount = parseFloat(raw);
    if (isNaN(newCount) || newCount <= 0) {
      setItems((prev) => prev.map((item, idx) => idx === i ? { ...item, count: null } : item));
      return;
    }
    const base = baseItems[i];
    const originalCount = base.count ?? 1;
    const ratio = newCount / originalCount;
    setItems((prev) => prev.map((item, idx) =>
      idx === i ? {
        ...item,
        count:     newCount,
        quantity:  base.quantity != null  ? Math.round(base.quantity  * ratio) : item.quantity,
        calories:  base.calories != null  ? Math.round(base.calories  * ratio * 10) / 10 : item.calories,
        protein_g: base.protein_g != null ? Math.round(base.protein_g * ratio * 10) / 10 : item.protein_g,
        carbs_g:   base.carbs_g != null   ? Math.round(base.carbs_g   * ratio * 10) / 10 : item.carbs_g,
        fats_g:    base.fats_g != null    ? Math.round(base.fats_g    * ratio * 10) / 10 : item.fats_g,
      } : item
    ));
    setTouched(true);
  };

  const totalCal  = items.reduce((s, m) => s + (m.calories  ?? 0), 0);
  const totalPro  = items.reduce((s, m) => s + (m.protein_g ?? 0), 0);
  const totalCarb = items.reduce((s, m) => s + (m.carbs_g   ?? 0), 0);
  const totalFat  = items.reduce((s, m) => s + (m.fats_g    ?? 0), 0);

  const handleConfirm = () => {
    setSaving(true);
    onConfirm(items);
  };

  return (
    <View style={cs.foodCard}>
      {/* ── Header ── */}
      <View style={cs.foodHeader}>
        <View style={cs.foodIcon}>
          <Ionicons name="nutrition" size={18} color={COLORS.primary} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={cs.mealLabel}>{mealLabel.toUpperCase()}</Text>
          <Text style={cs.foodName} numberOfLines={1}>
            {items.map((m) => m.food_name).join(" + ")}
          </Text>
        </View>
        <Text style={cs.foodTime}>{time}</Text>
      </View>

      {/* ── Per-item editable macro tables ── */}
      {items.map((item, i) => (
        <View key={i} style={[cs.foodItemBlock, i > 0 && cs.foodItemBlockDivider]}>
          {items.length > 1 && (
            <Text style={cs.foodItemBlockName} numberOfLines={1}>{item.food_name}</Text>
          )}
          <View style={cs.macroGrid}>
            <MacroCell
              label="CALORIES" unit="kcal"
              value={item.calories != null ? String(Math.round(item.calories)) : ""}
              onChange={(v) => updateItem(i, "calories", v)}
              onTouch={() => setTouched(true)}
            />
            <MacroCell
              label="PROTEIN" unit="g" accent
              value={item.protein_g != null ? String(Math.round(item.protein_g)) : ""}
              onChange={(v) => updateItem(i, "protein_g", v)}
              onTouch={() => setTouched(true)}
            />
            <MacroCell
              label="CARBS" unit="g"
              value={item.carbs_g != null ? String(Math.round(item.carbs_g)) : ""}
              onChange={(v) => updateItem(i, "carbs_g", v)}
              onTouch={() => setTouched(true)}
            />
            <MacroCell
              label="FAT" unit="g"
              value={item.fats_g != null ? String(Math.round(item.fats_g)) : ""}
              onChange={(v) => updateItem(i, "fats_g", v)}
              onTouch={() => setTouched(true)}
            />
          </View>
          {/* Count row (e.g. "4 eggs"), shown when AI returned a countable unit */}
          {item.count != null && item.unit != null ? (
            <View style={cs.quantityRow}>
              <Text style={cs.quantityLabel}>{item.unit.charAt(0).toUpperCase() + item.unit.slice(1)}s</Text>
              <TextInput
                style={cs.quantityInput}
                value={item.count != null ? String(item.count) : ""}
                onChangeText={(v) => updateCount(i, v)}
                keyboardType="decimal-pad"
                selectTextOnFocus
                placeholderTextColor="#3A3A3A"
              />
              <Text style={cs.quantityUnit}>{item.unit}s</Text>
            </View>
          ) : item.quantity != null ? (
            <View style={cs.quantityRow}>
              <Text style={cs.quantityLabel}>Quantity</Text>
              <TextInput
                style={cs.quantityInput}
                value={item.quantity != null ? String(Math.round(item.quantity)) : ""}
                onChangeText={(v) => updateItem(i, "quantity", v)}
                keyboardType="decimal-pad"
                selectTextOnFocus
                placeholderTextColor="#3A3A3A"
              />
              <Text style={cs.quantityUnit}>g</Text>
            </View>
          ) : null}
        </View>
      ))}

      {/* ── Totals row (multi-item only) ── */}
      {items.length > 1 && (
        <View style={cs.totalsRow}>
          <Text style={cs.totalsLabel}>TOTAL</Text>
          <Text style={cs.totalsValue}>{Math.round(totalCal)} kcal</Text>
          <Text style={cs.totalsDot}>·</Text>
          <Text style={[cs.totalsValue, { color: COLORS.primary }]}>{Math.round(totalPro)}g P</Text>
          <Text style={cs.totalsDot}>·</Text>
          <Text style={cs.totalsValue}>{Math.round(totalCarb)}g C</Text>
          <Text style={cs.totalsDot}>·</Text>
          <Text style={cs.totalsValue}>{Math.round(totalFat)}g F</Text>
        </View>
      )}

      {/* ── Status pill ── */}
      <View style={cs.statusPill}>
        <Ionicons name={touched ? "create-outline" : "checkmark-circle"} size={14} color={touched ? "#aaa" : COLORS.primary} />
        <Text style={cs.statusPillTxt}>
          {touched ? "Edited · tap confirm to log" : items.length > 1 ? `${items.length} items · from food database` : "Matched from food database"}
        </Text>
      </View>

      {/* ── Confirm / Cancel ── */}
      <PressableScale
        haptics="none" style={cs.confirmBtn} onPress={() => { haptic.light(); handleConfirm(); }} disabled={saving}>
        {saving ? (
          <ActivityIndicator color="#000" size="small" />
        ) : (
          <Text style={cs.confirmBtnTxt}>Confirm &amp; Log</Text>
        )}
      </PressableScale>
      <TouchableOpacity style={cs.cancelTextBtn} onPress={() => { haptic.select(); onCancel(); }} disabled={saving} activeOpacity={0.7}>
        <Text style={cs.cancelTextBtnTxt}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
}

const GOAL_LABEL: Record<string, string> = {
  lose: "Fat Loss",
  gain: "Muscle Gain",
  maintain: "Maintenance",
  strength: "Build Strength",
  fitness: "Stay Fit",
};

function shortDayFocus(day: ProgramProposalData["days"][0]): string {
  if (day.is_rest_day) return "Rest";
  const t = day.title.toLowerCase();
  // Structural terms FIRST. "Upper A - Push & Pull Power" is a correctly named
  // upper day, but testing "push" before "upper" labelled its tab "Push" — the
  // pill then contradicted the heading directly beneath it, and read as the
  // model having built a push day full of rows. It had not; this ordering was
  // the whole bug.
  if (t.includes("full")) return "Full";
  if (t.includes("upper")) return "Upper";
  if (t.includes("leg") || t.includes("lower")) return "Legs";
  if (t.includes("push") || t.includes("chest")) return "Push";
  if (t.includes("pull") || t.includes("back")) return "Pull";
  if (t.includes("shoulder")) return "Shldr";
  if (t.includes("arm") || t.includes("bicep") || t.includes("tricep")) return "Arms";
  if (t.includes("core") || t.includes("abs")) return "Core";
  if (t.includes("cardio") || t.includes("hiit")) return "Cardio";
  return "Train";
}

function guessMuscle(name: string): string {
  const n = name.toLowerCase();
  if (n.includes("squat") || n.includes("leg press") || n.includes("lunge")) return "Quads";
  if (n.includes("romanian") || n.includes("rdl") || n.includes("hamstring") || n.includes("leg curl")) return "Hamstrings";
  if (n.includes("calf") || n.includes("calves") || n.includes("calf raise")) return "Calves";
  if (n.includes("bench") || n.includes(" fly") || n.includes("chest") || n.includes("push-up") || n.includes("push up")) return "Chest";
  if (n.includes("pull-up") || n.includes("pull up") || n.includes("lat pull") || n.includes(" row")) return "Back";
  if (n.includes("overhead") || n.includes("shoulder") || n.includes("lateral raise") || n.includes("front raise")) return "Shoulders";
  if (n.includes("curl") && !n.includes("leg")) return "Biceps";
  if (n.includes("tricep") || n.includes("extension") || n.includes("pushdown") || n.includes("skull")) return "Triceps";
  if (n.includes("plank") || n.includes("crunch") || n.includes("core")) return "Core";
  if (n.includes("hip thrust") || n.includes("glute bridge")) return "Glutes";
  return "";
}

function ProgramProposalCard({
  data,
  onActivate,
  onAddToPlan,
  onCancel,
  onSwap,
  getAlternatives,
  frozen = false,
}: {
  data: ProgramProposalData;
  onActivate?: () => void;
  onAddToPlan?: () => Promise<void> | void;
  onCancel?: () => void;
  onSwap?: (updated: ProgramProposalData) => Promise<void> | void;
  getAlternatives?: (exerciseName: string) => Promise<Array<{ name: string; muscle: string }>>;
  frozen?: boolean;
}) {
  const trainingDays = data.days.filter((d) => !d.is_rest_day);
  const [busy, setBusy] = useState<null | "activate" | "add">(null);
  const [selectedIdx, setSelectedIdx] = useState(() => {
    const i = data.days.findIndex((d) => !d.is_rest_day);
    return i < 0 ? 0 : i;
  });
  // Which exercise (index within the selected day) is showing swap options.
  const [swap, setSwap] = useState<{ exIndex: number; loading: boolean; options: Array<{ name: string; muscle: string }> } | null>(null);

  const selectedDay = data.days[selectedIdx] ?? data.days[0];
  const dayExercises = selectedDay?.exercises ?? [];
  const totalEx = trainingDays.reduce((n, d) => n + (d.exercises?.length ?? 0), 0);
  const avgEx = trainingDays.length ? Math.round(totalEx / trainingDays.length) : 0;
  // Estimated from what the plan actually prescribes. The old version was a
  // lookup on exercise count alone — it could not see sets or rest, so six
  // exercises always read "~60 min" whether that was six singles at 60s rest
  // or 4x12 at 120s. A real 72-minute day was labelled 60 and looked like it
  // had overrun a budget it was in fact inside.
  // 40s of working time per set is the figure src/lib/starterPlan.ts uses.
  const WORK_S = 40;
  const WARMUP_S = 8 * 60;
  const sessionMin = trainingDays.length
    ? Math.round(
        trainingDays.reduce((tot, d) => {
          const work = (d.exercises ?? []).reduce(
            (t, ex) => t + (ex.sets || 3) * (WORK_S + (ex.rest_seconds ?? 90)),
            0
          );
          return tot + WARMUP_S + work;
        }, 0) /
          trainingDays.length /
          60
      )
    : 45;
  const canSwap = !frozen && !!onSwap;

  const selectDay = (idx: number) => { haptic.select(); setSwap(null); setSelectedIdx(idx); };

  const openSwap = async (exIndex: number, exName: string) => {
    if (swap?.exIndex === exIndex) { setSwap(null); return; }
    haptic.light();
    setSwap({ exIndex, loading: true, options: [] });
    try {
      const opts = (await getAlternatives?.(exName)) ?? [];
      const filtered = opts.filter((o) => o.name.toLowerCase() !== exName.toLowerCase()).slice(0, 4);
      setSwap({ exIndex, loading: false, options: filtered });
    } catch {
      setSwap({ exIndex, loading: false, options: [] });
    }
  };

  const applySwap = (exIndex: number, newName: string) => {
    haptic.medium();
    const newDays = data.days.map((d) =>
      d.day_number === selectedDay.day_number
        ? { ...d, exercises: (d.exercises ?? []).map((e, ei) => (ei === exIndex ? { ...e, name: newName } : e)) }
        : d
    );
    setSwap(null);
    onSwap?.({ ...data, days: newDays });
  };

  const doAdd = async () => {
    if (busy) return;
    haptic.medium();
    setBusy("add");
    try { await onAddToPlan?.(); } finally { setBusy(null); }
  };

  return (
    <View style={[pc.card, frozen && pc.cardFrozen]}>
      <View style={pc.accent} />

      {/* Header row: PROGRAM tag + freq badge */}
      <View style={pc.topRow}>
        <View style={pc.programTag}>
          <Ionicons name="barbell" size={11} color={COLORS.primary} />
          <Text style={pc.programTagTxt}>PROGRAM</Text>
        </View>
        <View style={pc.freqBadge}>
          <Text style={pc.freqBadgeTxt}>{trainingDays.length}×/wk</Text>
        </View>
      </View>

      {/* Title + subtitle */}
      <Text style={pc.title}>{data.title}</Text>
      <Text style={pc.subtitle}>
        {GOAL_LABEL[data.goal] ?? data.goal} · ~{sessionMin} min per session
      </Text>

      {/* Day pills, tap to preview that day */}
      <GestureDetector gesture={Gesture.Native().blocksExternalGesture(tabSwipeRef)}>
      <ScrollView keyboardDismissMode={KEYBOARD_DISMISS}
        horizontal
        showsHorizontalScrollIndicator={false}
        decelerationRate="fast"
        style={pc.pillsScroll}
        contentContainerStyle={pc.pillsContent}
      >
        {data.days.map((day, idx) => {
          const isRest = day.is_rest_day;
          const selected = idx === selectedIdx;
          return (
            <TouchableOpacity
              key={day.day_number}
              onPress={() => selectDay(idx)}
              activeOpacity={0.8}
              style={[pc.dayPill, !isRest && !selected && pc.dayPillTraining, selected && pc.dayPillActive]}
            >
              <Text style={[pc.dayPillNum, selected && pc.dayPillNumActive]}>D{day.day_number}</Text>
              <Text style={[pc.dayPillSub, selected && pc.dayPillSubActive]}>
                {isRest ? "Rest" : shortDayFocus(day)}
              </Text>
            </TouchableOpacity>
          );
        })}
      </ScrollView>
      </GestureDetector>

      {/* Selected day heading */}
      <Text style={pc.dayHeading}>
        Day {selectedDay?.day_number} · <Text style={pc.dayHeadingFocus}>{selectedDay?.title}</Text>
      </Text>

      {/* Selected day content */}
      {selectedDay?.is_rest_day ? (
        <View style={pc.restBox}>
          <Ionicons name="bed-outline" size={18} color="#555" />
          <Text style={pc.restTxt}>Rest & recover, no session today.</Text>
        </View>
      ) : (
        dayExercises.map((ex, i) => {
          const muscle = guessMuscle(ex.name);
          const open = swap?.exIndex === i;
          return (
            <View key={`${selectedDay.day_number}-${i}`}>
              <View style={pc.exRow}>
                <View style={pc.exNum}><Text style={pc.exNumTxt}>{i + 1}</Text></View>
                <View style={pc.exInfo}>
                  <Text style={pc.exName}>{ex.name}</Text>
                  {!!muscle && <Text style={pc.exMuscle}>{muscle}</Text>}
                </View>
                <Text style={pc.exSets}>{ex.sets} × {ex.reps.split("/")[0]}</Text>
                {canSwap && (
                  <TouchableOpacity
                    style={[pc.swapBtn, open && pc.swapBtnActive]}
                    onPress={() => openSwap(i, ex.name)}
                    activeOpacity={0.8}
                    hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
                  >
                    <Ionicons name="swap-horizontal" size={16} color={open ? "#000" : COLORS.primary} />
                  </TouchableOpacity>
                )}
              </View>

              {open && (
                <View style={pc.swapPanel}>
                  {swap!.loading ? (
                    <ActivityIndicator color={COLORS.primary} size="small" style={{ paddingVertical: 8 }} />
                  ) : swap!.options.length === 0 ? (
                    <Text style={pc.swapEmpty}>No similar exercises found.</Text>
                  ) : (
                    <>
                      <Text style={pc.swapLabel}>Swap for a similar move</Text>
                      {swap!.options.map((opt) => (
                        <TouchableOpacity
                          key={opt.name}
                          style={pc.swapOpt}
                          onPress={() => applySwap(i, opt.name)}
                          activeOpacity={0.8}
                        >
                          <Ionicons name="repeat" size={15} color={COLORS.primary} />
                          <View style={pc.swapOptInfo}>
                            <Text style={pc.swapOptName}>{opt.name}</Text>
                            {!!opt.muscle && <Text style={pc.swapOptMuscle}>{opt.muscle}</Text>}
                          </View>
                        </TouchableOpacity>
                      ))}
                    </>
                  )}
                </View>
              )}
            </View>
          );
        })
      )}

      {/* Buttons, hidden when frozen */}
      {!frozen && (
        <>
          <View style={pc.notice}>
            <View style={pc.noticeDot} />
            <Text style={pc.noticeTxt}>
              “Make current” replaces your active program. “Add” saves this to My Plans and keeps your current one.
            </Text>
          </View>
          <TouchableOpacity
            style={pc.activateBtn}
            onPress={() => { if (busy) return; haptic.medium(); setBusy("activate"); onActivate?.(); }}
            disabled={!!busy}
            activeOpacity={0.85}
          >
            {busy === "activate" ? (
              <ActivityIndicator color="#000" size="small" />
            ) : (
              <Text style={pc.activateTxt}>{data.isModify ? "Update program" : "Make this current plan"}</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity style={pc.secondaryBtn} onPress={doAdd} disabled={!!busy} activeOpacity={0.85}>
            {busy === "add" ? (
              <ActivityIndicator color={COLORS.primary} size="small" />
            ) : (
              <Text style={pc.secondaryTxt}>Add to workout plan</Text>
            )}
          </TouchableOpacity>
          <TouchableOpacity style={pc.cancelBtn} onPress={() => { haptic.select(); onCancel?.(); }} disabled={!!busy} activeOpacity={0.75}>
            <Text style={pc.cancelTxt}>Cancel</Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

const pc = StyleSheet.create({
  card: {
    backgroundColor: "#0D0D0F",
    borderRadius: 24,
    padding: 20,
    paddingLeft: 24,
    marginTop: 8,
    alignSelf: "stretch",
    borderWidth: 1,
    borderColor: "rgba(216,255,42,0.18)",
    overflow: "hidden",
  },
  // Intentionally identical to the live card. Dimming to 0.9 and swapping the
  // lime border for grey made a program you had just had built look faded the
  // moment it scrolled into history. It is already unmistakably inert — every
  // button is stripped when frozen — so the dimming bought no signal.
  cardFrozen: {},
  accent: {
    position: "absolute", left: 0, top: 0, bottom: 0,
    width: 3.5, backgroundColor: COLORS.primary, borderRadius: 2,
  },
  topRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between", marginBottom: 10 },
  programTag: { flexDirection: "row", alignItems: "center", gap: 5 },
  programTagTxt: { color: COLORS.primary, fontFamily: FONTS.bold, fontSize: 11, letterSpacing: 1.2 },
  freqBadge: { backgroundColor: COLORS.primary, borderRadius: 12, paddingHorizontal: 10, paddingVertical: 4 },
  freqBadgeTxt: { color: "#000", fontFamily: FONTS.black, fontSize: 12 },
  title: { color: "#FFF", fontFamily: FONTS.black, fontSize: 22, letterSpacing: -0.5, lineHeight: 26, marginBottom: 4 },
  subtitle: { color: "#666", fontFamily: FONTS.medium, fontSize: 13, marginBottom: 16 },
  pillsScroll: { marginBottom: 18 },
  pillsContent: { gap: 8, flexDirection: "row", paddingVertical: 2 },
  dayPill: {
    backgroundColor: "#1A1A1A", borderRadius: 12,
    paddingHorizontal: 10, paddingVertical: 8,
    alignItems: "center", minWidth: 48,
    borderWidth: 1, borderColor: "transparent",
  },
  dayPillTraining: { borderColor: "rgba(216,255,42,0.35)" },
  dayPillActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  dayPillNum: { color: "#DDD", fontFamily: FONTS.black, fontSize: 12 },
  dayPillNumActive: { color: "#000" },
  dayPillSub: { color: "#666", fontFamily: FONTS.medium, fontSize: 10, marginTop: 2 },
  dayPillSubActive: { color: "#000" },
  dayHeading: { color: "#888", fontFamily: FONTS.semibold, fontSize: 12.5, letterSpacing: 0.3, marginBottom: 2, textTransform: "uppercase" },
  dayHeadingFocus: { color: "#FFF", fontFamily: FONTS.bold },
  restBox: {
    flexDirection: "row", alignItems: "center", gap: 10,
    paddingVertical: 16, marginTop: 6,
  },
  restTxt: { color: "#666", fontFamily: FONTS.medium, fontSize: 13.5 },
  exRow: {
    flexDirection: "row", alignItems: "center", gap: 12,
    paddingVertical: 11,
    borderTopWidth: 1, borderTopColor: "#131313",
  },
  exNum: {
    width: 30, height: 30, borderRadius: 9,
    backgroundColor: "#141620", borderWidth: 1, borderColor: "#1E2230",
    alignItems: "center", justifyContent: "center", flexShrink: 0,
  },
  exNumTxt: { color: "#5B80FF", fontFamily: FONTS.bold, fontSize: 13 },
  exInfo: { flex: 1, minWidth: 0 },
  exName: { color: "#FFF", fontFamily: FONTS.bold, fontSize: 14.5, letterSpacing: -0.2 },
  exMuscle: { color: "#555", fontFamily: FONTS.regular, fontSize: 12, marginTop: 1 },
  exSets: { color: COLORS.primary, fontFamily: FONTS.bold, fontSize: 13.5 },
  swapBtn: {
    width: 32, height: 32, borderRadius: 10, flexShrink: 0,
    backgroundColor: "rgba(216,255,42,0.12)",
    alignItems: "center", justifyContent: "center",
    borderWidth: 1, borderColor: "rgba(216,255,42,0.25)",
  },
  swapBtnActive: { backgroundColor: COLORS.primary, borderColor: COLORS.primary },
  swapPanel: {
    backgroundColor: "#111214", borderRadius: 14, padding: 8,
    marginTop: 2, marginBottom: 8, marginLeft: 42,
    borderWidth: 1, borderColor: "#1C1D20",
  },
  swapLabel: { color: "#666", fontFamily: FONTS.semibold, fontSize: 11, letterSpacing: 0.4, textTransform: "uppercase", paddingHorizontal: 6, paddingTop: 4, paddingBottom: 6 },
  swapEmpty: { color: "#666", fontFamily: FONTS.regular, fontSize: 13, padding: 10 },
  swapOpt: {
    flexDirection: "row", alignItems: "center", gap: 10,
    paddingVertical: 10, paddingHorizontal: 8, borderRadius: 10,
  },
  swapOptInfo: { flex: 1, minWidth: 0 },
  swapOptName: { color: "#EFEFEF", fontFamily: FONTS.semibold, fontSize: 14 },
  swapOptMuscle: { color: "#555", fontFamily: FONTS.regular, fontSize: 11.5, marginTop: 1 },
  notice: {
    flexDirection: "row", alignItems: "flex-start", gap: 8,
    backgroundColor: "#131313", borderRadius: 12,
    paddingHorizontal: 14, paddingVertical: 10,
    marginTop: 16, marginBottom: 14,
  },
  noticeDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: "#555", flexShrink: 0, marginTop: 5 },
  noticeTxt: { color: "#666", fontFamily: FONTS.regular, fontSize: 12, flex: 1, lineHeight: 17 },
  activateBtn: {
    height: 52, borderRadius: 16,
    backgroundColor: COLORS.primary, justifyContent: "center", alignItems: "center",
  },
  activateTxt: { color: "#000", fontFamily: FONTS.bold, fontSize: 15 },
  secondaryBtn: {
    height: 50, borderRadius: 16, marginTop: 10,
    backgroundColor: "#17190F", justifyContent: "center", alignItems: "center",
    borderWidth: 1, borderColor: "rgba(216,255,42,0.3)",
  },
  secondaryTxt: { color: COLORS.primary, fontFamily: FONTS.bold, fontSize: 14.5 },
  cancelBtn: { height: 44, justifyContent: "center", alignItems: "center", marginTop: 4 },
  cancelTxt: { color: "#777", fontFamily: FONTS.medium, fontSize: 14 },
});

type WizardQuestion = { q: string; opts: string[] };

// The plan brief replaces the old four-question program wizard, which asked
// things the coach already knows: primary_goal, available_days_per_week and
// session_duration_mins all live in user_preferences and ride the context, and
// "Full Body / Upper / Lower / Arms & Core" is coarser than the muscle taxonomy
// the exercise filter actually uses. Re-asking known facts reads as not having
// listened. These two are genuinely unknowable from stored data: a static
// injuries field cannot say what hurts THIS week, and nothing in the schema
// captures a deadline.
// Tapping this does not answer the question — it opens a text field inside the
// card. It used to be sent as the literal answer, so the coach received
// "Currently bothering me: Something else" and had no choice but to reply
// "what's the something else?" — a card whose whole point was to avoid a
// back-and-forth caused one, and the typing happened in the message box where
// the answer was no longer attached to the brief.
const BRIEF_FREE_TEXT_OPT = "Something else";

// The short label shown in the user's bubble, in place of the long sentence
// the brief composes for the model. Persisted with the brief so a reloaded
// thread reads identically to the live one.
const BRIEF_DISPLAY = "Build me a program";

// Survives tab switches and app restarts so a half-typed message is not lost.
const COACH_DRAFT_KEY = "coach_draft";

// "You've reached the beginning of this conversation" is only information if
// there was something to scroll back through. On a fresh chat it hung that
// caption above the first two messages, in an otherwise empty screen.
const HISTORY_END_MIN = 10;

const PLAN_BRIEF_MCQ: WizardQuestion[] = [
  {
    q: "Anything bothering you right now?",
    opts: ["Nothing", "Shoulder", "Lower back", "Knee", BRIEF_FREE_TEXT_OPT],
  },
  {
    q: "Working towards a date?",
    opts: ["Not really", "~4 weeks", "~8 weeks", "12+ weeks"],
  },
];

// Two muscle-map steps, then the questions above.
const BRIEF_MUSCLE_STEPS = 2;
const BRIEF_TOTAL_STEPS = BRIEF_MUSCLE_STEPS + PLAN_BRIEF_MCQ.length;

type PlanBriefState = {
  step: number;
  focus: string[];
  avoid: string[];
  answers: (string | null)[];
  generating: boolean;
  // Non-null while the free-text field is open on the current step. "" is a
  // meaningful value (field open, nothing typed yet), so this is not a boolean.
  freeText?: string | null;
} | null;

// Asked on its own once the user has already tapped the exact exercise — the
// muscle is then already known, so asking for it again is pure friction.
const SWAP_REASON_QUESTION: WizardQuestion = {
  q: "Why are you swapping it?",
  opts: ["No equipment", "Too difficult", "Need variety", "Minor discomfort"],
};

const SWAP_WIZARD_QUESTIONS: WizardQuestion[] = [
  {
    // Individual muscles, not lumped pairs: "Back & Biceps" hid triceps entirely
    // and forced unrelated muscles into one button. These match the muscle tags
    // on the exercises table, so the coach can filter to real alternatives.
    q: "Which muscle is it for?",
    opts: [
      "Chest", "Back", "Shoulders", "Biceps", "Triceps",
      "Quads", "Hamstrings", "Glutes", "Calves", "Core",
    ],
  },
  SWAP_REASON_QUESTION,
];

/**
 * Order DB alternatives by the user's stated reason. The list already comes back
 * ranked by similarity; this just re-weights it so "No equipment" surfaces
 * home-friendly options, "Too difficult" surfaces easier ones, etc.
 */
function rankAlternatives(list: Exercise[], reason: string): Exercise[] {
  const easier = (d?: string) =>
    (d || "").toLowerCase() === "beginner" ? 2 : (d || "").toLowerCase() === "intermediate" ? 1 : 0;
  const homeable = (e?: string) =>
    /bodyweight|band|dumbbell|kettlebell|none/i.test(e || "") ? 2 : /machine|cable/i.test(e || "") ? 1 : 0;
  const gentle = (e?: string) => (/machine|cable|band/i.test(e || "") ? 2 : /barbell/i.test(e || "") ? 0 : 1);

  const score = (ex: Exercise): number => {
    switch (reason) {
      case "No equipment":
        return homeable(ex.equipment) * 3 + easier(ex.difficulty);
      case "Too difficult":
        return easier(ex.difficulty) * 3 + (ex.is_compound ? 0 : 2);
      case "Minor discomfort":
        // Prefer guided/lower-load patterns that are easier on a sore joint.
        return gentle(ex.equipment) * 3 + easier(ex.difficulty);
      default:
        return 0; // "Need variety" — keep the similarity order as-is
    }
  };
  return [...list].sort((a, b) => score(b) - score(a));
}

const ADJUST_WIZARD_QUESTIONS: WizardQuestion[] = [
  {
    q: "What's off about your plan?",
    opts: ["Too hard", "Too easy", "Hit a plateau", "Change focus", "Takes too long"],
  },
];

// Each answer maps to a sharp, grounded prompt so the coach gives an instant
// targeted fix instead of asking "what do you mean?".
const ADJUST_REASON_PROMPTS: Record<string, string> = {
  "Too hard":
    "My current program is beating me up, the weights or volume feel too much. Suggest 2-3 specific changes to make it manageable while still progressing. Be direct, no questions.",
  "Too easy":
    "My current program feels too easy, I'm not challenged. Suggest 2-3 specific ways to raise the difficulty (weight, volume, or intensity). Be direct, no questions.",
  "Hit a plateau":
    "I've stalled on my current program. Based on my recent training, give me 2-3 specific changes to break the plateau. Be direct, no questions.",
  "Change focus":
    "I want to change what my program emphasizes. Give me 2-3 concrete ways to reweight it (e.g. toward upper body, strength, or athleticism) and say which fits my goal. Be direct.",
  "Takes too long":
    "My workouts take too long. Suggest 2-3 specific ways to cut session time (supersets, trimming junk volume, etc.) without losing results. Be direct, no questions.",
};

// A cheat meal isn't one scenario — lunch out on a cut and a late night out on
// a bulk need opposite plans. These three answers are the only things the coach
// can't already see: today's intake, the targets and the goal are in its
// context, so asking about them again would be friction for nothing.
const CHEAT_WIZARD_QUESTIONS: WizardQuestion[] = [
  {
    q: "Which meal are you cheating?",
    opts: ["Lunch", "Dinner", "Night out", "Whole day"],
  },
  {
    q: "What's it going to be?",
    opts: ["Restaurant meal", "Fast food", "Drinks + snacks", "Dessert", "Home feast"],
  },
  {
    q: "How hard are you going?",
    opts: ["Keep it tight", "Balanced", "Full send"],
  },
];

type WizardState = {
  step: number;
  answers: (string | null)[];
  generating: boolean;
} | null;

function WizardCard({
  wizard,
  questions,
  generatingLabel,
  onSelect,
}: {
  wizard: NonNullable<WizardState>;
  questions: WizardQuestion[];
  generatingLabel?: string;
  onSelect: (option: string) => void;
}) {
  const { step, generating } = wizard;
  const q = questions[step];

  if (generating) {
    return (
      <View style={wc.card}>
        <View style={wc.accent} />
        <ActivityIndicator color={COLORS.primary} style={{ marginBottom: 14 }} />
        <Text style={wc.genTxt}>{generatingLabel ?? "Working on it..."}</Text>
        <Text style={wc.genSub}>This takes a few seconds</Text>
      </View>
    );
  }

  return (
    <View style={wc.card}>
      <View style={wc.accent} />
      <View style={wc.progressRow}>
        {questions.map((_, i) => (
          <View key={i} style={[wc.progressDot, i <= step && wc.progressDotActive]} />
        ))}
      </View>
      <Text style={wc.stepTxt}>Step {step + 1} of {questions.length}</Text>
      <Text style={wc.question}>{q.q}</Text>
      <View style={wc.optionsGrid}>
        {q.opts.map((opt) => (
          <TouchableOpacity key={opt} style={wc.option} onPress={() => { haptic.select(); onSelect(opt); }} activeOpacity={0.8}>
            <Text style={wc.optionTxt}>{opt}</Text>
          </TouchableOpacity>
        ))}
      </View>
    </View>
  );
}

// What the user actually chose, kept so the brief stays visible in the thread
// instead of collapsing into the prompt it produced.
type PlanBriefSummary = {
  focus: string[];
  avoid: string[];
  pain?: string | null;
  deadline?: string | null;
  // What the user's bubble should read. The message actually SENT is the long
  // composed sentence, because that is what the model needs; the bubble shows
  // this. It rides in the persisted brief so a reloaded thread renders the same
  // three words the live one did instead of the machine-ish sentence.
  display?: string;
};

// The brief, frozen. Sending the composed prompt as a plain user bubble meant a
// wall of machine-ish text where a moment earlier there had been a body map —
// the choices were still there, but nothing showed you had made them.
function BriefSummaryCard({ data }: { data: PlanBriefSummary }) {
  const rows: { label: string; items: string[]; avoid?: boolean }[] = [];
  if (data.focus.length) rows.push({ label: "Focus", items: data.focus });
  if (data.avoid.length) rows.push({ label: "Go easy on", items: data.avoid, avoid: true });

  const notes: string[] = [];
  if (data.pain && data.pain !== "Nothing") notes.push(data.pain);
  if (data.deadline && data.deadline !== "Not really") notes.push(data.deadline);

  if (!rows.length && !notes.length) return null;

  return (
    <View style={bs.card}>
      <Text style={bs.title}>Your brief</Text>
      {rows.map((r) => (
        <View key={r.label} style={bs.row}>
          <Text style={bs.rowLabel}>{r.label}</Text>
          <View style={bs.chips}>
            {r.items.map((m) => (
              <View key={m} style={[bs.chip, r.avoid && bs.chipAvoid]}>
                <Text style={[bs.chipTxt, r.avoid && bs.chipTxtAvoid]}>{m}</Text>
              </View>
            ))}
          </View>
        </View>
      ))}
      {notes.length > 0 && (
        <View style={bs.row}>
          <Text style={bs.rowLabel}>Noted</Text>
          <View style={bs.chips}>
            {notes.map((n) => (
              <View key={n} style={bs.chipMuted}>
                <Text style={bs.chipTxtMuted}>{n}</Text>
              </View>
            ))}
          </View>
        </View>
      )}
    </View>
  );
}

const bs = StyleSheet.create({
  card: {
    alignSelf: "flex-end", maxWidth: "88%",
    backgroundColor: COLORS.surfaceElevated,
    borderRadius: 18, borderWidth: 1, borderColor: COLORS.border,
    padding: 14, marginBottom: 10,
  },
  title: { color: COLORS.textMuted, fontFamily: FONTS.bold, fontSize: 10, letterSpacing: 1.4, textTransform: "uppercase", marginBottom: 10 },
  row: { marginBottom: 8 },
  rowLabel: { color: COLORS.textFaint, fontFamily: FONTS.semibold, fontSize: 11, marginBottom: 5 },
  chips: { flexDirection: "row", flexWrap: "wrap", gap: 6 },
  chip: { backgroundColor: "rgba(216,255,42,0.12)", borderWidth: 1, borderColor: "rgba(216,255,42,0.30)", borderRadius: 999, paddingHorizontal: 10, paddingVertical: 5 },
  chipAvoid: { backgroundColor: "rgba(255,107,107,0.12)", borderColor: "rgba(255,107,107,0.30)" },
  chipTxt: { color: COLORS.primary, fontFamily: FONTS.semibold, fontSize: 12 },
  chipTxtAvoid: { color: "#FF6B6B" },
  chipMuted: { backgroundColor: "rgba(255,255,255,0.06)", borderRadius: 999, paddingHorizontal: 10, paddingVertical: 5 },
  chipTxtMuted: { color: COLORS.text, fontFamily: FONTS.medium, fontSize: 12 },
});

// Program generation runs on a thinking model with an 8000-token budget and
// takes tens of seconds, not "a few". A spinner that promises seconds and then
// hangs for half a minute reads as broken, and people back out and retry —
// which starts a second generation and makes it worse.
//
// These stages are honest about what is actually happening in
// _handle_tool_intent, and they CLAMP at the last one rather than looping.
// Looping back to "Reading your profile" after 40s would say the opposite of
// what it means, that nothing is progressing.
const BRIEF_STAGES = [
  "Reading your profile...",
  "Picking exercises for your goal...",
  "Working around your injuries...",
  "Balancing the week...",
  "Writing up sets and reps...",
];
// 9s, not 6. Five stages across a 30-45s build means the text should feel
// like it is tracking real work, not ticking through a list.
const BRIEF_STAGE_MS = 9000;

function BriefGenerating() {
  const [stage, setStage] = useState(0);
  useEffect(() => {
    const id = setInterval(
      () => setStage((n) => Math.min(n + 1, BRIEF_STAGES.length - 1)),
      BRIEF_STAGE_MS
    );
    return () => clearInterval(id);
  }, []);

  return (
    <View style={wc.card}>
      <View style={wc.accent} />
      <ActivityIndicator color={COLORS.primary} style={{ marginBottom: 14 }} />
      <Text style={wc.genTxt}>{BRIEF_STAGES[stage]}</Text>
      <Text style={wc.genSub}>Building a full program takes a moment</Text>
      <View style={pbg.track}>
        {BRIEF_STAGES.map((_, i) => (
          <View key={i} style={[pbg.seg, i <= stage && pbg.segOn]} />
        ))}
      </View>
    </View>
  );
}

const pbg = StyleSheet.create({
  track: { flexDirection: "row", gap: 4, marginTop: 14, width: "100%" },
  seg: { flex: 1, height: 3, borderRadius: 2, backgroundColor: "rgba(255,255,255,0.10)" },
  segOn: { backgroundColor: COLORS.primary },
});

// ─── Plan Brief Card ─────────────────────────────────────────────────────────
// Steps 0-1 are the same BodyMap the exercise filter and detail sheet use, in
// multi-select mode, so "prioritise" and "avoid" come back as real muscle slugs
// rather than a coarse bucket. Every step is skippable — a brief you cannot
// dismiss is a form, and nobody fills in a form to talk to their coach.
function PlanBriefCard({
  brief,
  gender,
  onToggleMuscle,
  onNext,
  onSkip,
  onSelect,
  onFreeText,
}: {
  brief: NonNullable<PlanBriefState>;
  gender: "male" | "female";
  onToggleMuscle: (slug: string) => void;
  onNext: () => void;
  onSkip: () => void;
  onSelect: (option: string) => void;
  // null closes the field and goes back to the options.
  onFreeText: (value: string | null) => void;
}) {
  const { step, generating } = brief;

  // Hooks cannot live behind a conditional, so the stage ticker is its own
  // component rather than inline state here.
  if (generating) return <BriefGenerating />;

  const isMuscleStep = step < BRIEF_MUSCLE_STEPS;
  const picking = step === 0 ? brief.focus : brief.avoid;

  return (
    <View style={wc.card}>
      <View style={wc.accent} />
      <View style={wc.progressRow}>
        {Array.from({ length: BRIEF_TOTAL_STEPS }).map((_, i) => (
          <View key={i} style={[wc.progressDot, i <= step && wc.progressDotActive]} />
        ))}
      </View>
      <Text style={wc.stepTxt}>Step {step + 1} of {BRIEF_TOTAL_STEPS}</Text>

      {isMuscleStep ? (
        <>
          <Text style={wc.question}>
            {step === 0 ? "Anything you want to prioritise?" : "Anything to go easy on?"}
          </Text>
          <Text style={pb.hint}>
            {step === 0
              ? "Tap the muscles you want more of. Optional."
              : "Tap anything sore, injured, or you would rather skip. Optional."}
          </Text>

          <View style={pb.mapWrap}>
            <BodyMap
              selectable
              selectedMuscles={picking}
              onSelectMuscle={onToggleMuscle}
              gender={gender}
            />
          </View>

          {picking.length > 0 && (
            <View style={pb.chipRow}>
              {picking.map((m) => (
                <TouchableOpacity
                  key={m}
                  style={[pb.chip, step === 1 && pb.chipAvoid]}
                  onPress={() => { haptic.select(); onToggleMuscle(m); }}
                  activeOpacity={0.8}
                >
                  <Text style={[pb.chipTxt, step === 1 && pb.chipTxtAvoid]}>
                    {MUSCLE_LABELS[m] ?? m}
                  </Text>
                  <Ionicons
                    name="close"
                    size={13}
                    color={step === 1 ? "#FF6B6B" : COLORS.primary}
                    style={{ marginLeft: 5 }}
                  />
                </TouchableOpacity>
              ))}
            </View>
          )}

          <View style={pb.actionRow}>
            <TouchableOpacity style={pb.skipBtn} onPress={() => { haptic.select(); onSkip(); }} activeOpacity={0.8}>
              <Text style={pb.skipTxt}>Skip</Text>
            </TouchableOpacity>
            <TouchableOpacity style={pb.nextBtn} onPress={() => { haptic.select(); onNext(); }} activeOpacity={0.85}>
              <Text style={pb.nextTxt}>
                {picking.length > 0 ? "Continue (" + picking.length + ")" : "Continue"}
              </Text>
            </TouchableOpacity>
          </View>
        </>
      ) : (
        <>
          <Text style={wc.question}>{PLAN_BRIEF_MCQ[step - BRIEF_MUSCLE_STEPS].q}</Text>
          {brief.freeText == null ? (
            <View style={wc.optionsGrid}>
              {PLAN_BRIEF_MCQ[step - BRIEF_MUSCLE_STEPS].opts.map((opt) => (
                <TouchableOpacity
                  key={opt}
                  style={wc.option}
                  onPress={() => { haptic.select(); onSelect(opt); }}
                  activeOpacity={0.8}
                >
                  <Text style={wc.optionTxt}>{opt}</Text>
                </TouchableOpacity>
              ))}
            </View>
          ) : (
            <View>
              <Text style={pb.hint}>Tell me what it is and I&apos;ll build around it.</Text>
              <TextInput
                style={pb.freeInput}
                value={brief.freeText}
                onChangeText={onFreeText}
                placeholder="e.g. right wrist, tight hips, recovering from flu"
                placeholderTextColor={COLORS.textFaint}
                autoFocus
                returnKeyType="done"
                maxLength={120}
                onSubmitEditing={() => {
                  const v = (brief.freeText ?? "").trim();
                  if (v) { haptic.select(); onSelect(v); }
                }}
              />
              <View style={pb.actionRow}>
                <TouchableOpacity
                  style={pb.skipBtn}
                  onPress={() => { haptic.select(); onFreeText(null); }}
                  activeOpacity={0.8}
                >
                  <Text style={pb.skipTxt}>Back</Text>
                </TouchableOpacity>
                <TouchableOpacity
                  style={[pb.nextBtn, !(brief.freeText ?? "").trim() && pb.nextBtnOff]}
                  disabled={!(brief.freeText ?? "").trim()}
                  onPress={() => {
                    const v = (brief.freeText ?? "").trim();
                    if (v) { haptic.select(); onSelect(v); }
                  }}
                  activeOpacity={0.85}
                >
                  <Text style={pb.nextTxt}>Continue</Text>
                </TouchableOpacity>
              </View>
            </View>
          )}
        </>
      )}
    </View>
  );
}

const pb = StyleSheet.create({
  hint: { color: COLORS.textMuted, fontFamily: FONTS.regular, fontSize: 12, marginTop: -6, marginBottom: 12 },
  mapWrap: { alignItems: "center", marginBottom: 12 },
  chipRow: { flexDirection: "row", flexWrap: "wrap", gap: 8, marginBottom: 14 },
  chip: {
    flexDirection: "row", alignItems: "center",
    backgroundColor: "rgba(216,255,42,0.12)",
    borderWidth: 1, borderColor: "rgba(216,255,42,0.35)",
    borderRadius: 999, paddingHorizontal: 11, paddingVertical: 6,
  },
  chipAvoid: { backgroundColor: "rgba(255,107,107,0.12)", borderColor: "rgba(255,107,107,0.35)" },
  chipTxt: { color: COLORS.primary, fontFamily: FONTS.semibold, fontSize: 12 },
  chipTxtAvoid: { color: "#FF6B6B" },
  actionRow: { flexDirection: "row", gap: 10 },
  skipBtn: {
    paddingHorizontal: 18, paddingVertical: 12, borderRadius: 12,
    backgroundColor: COLORS.surfaceElevated, borderWidth: 1, borderColor: COLORS.border,
  },
  skipTxt: { color: COLORS.textMuted, fontFamily: FONTS.semibold, fontSize: 13 },
  nextBtn: {
    flex: 1, alignItems: "center", justifyContent: "center",
    paddingVertical: 12, borderRadius: 12, backgroundColor: COLORS.primary,
  },
  nextTxt: { color: "#0A0A0A", fontFamily: FONTS.extrabold, fontSize: 14 },
  nextBtnOff: { opacity: 0.4 },
  freeInput: {
    backgroundColor: COLORS.surfaceElevated,
    borderWidth: 1, borderColor: COLORS.border, borderRadius: 12,
    paddingHorizontal: 14, paddingVertical: 12, marginBottom: 12,
    color: COLORS.text, fontFamily: FONTS.regular, fontSize: 14,
  },
});

// Backward-compat alias
type ProgramWizardState = WizardState;

// ─── Exercise Picker Card ────────────────────────────────────────────────────
// Shown before the swap wizard, lets the user tap the exercise they want to
// replace instead of letting the AI guess.

function ExercisePickerCard({
  exercises,
  fetching,
  onPick,
  onSkip,
}: {
  exercises: string[];
  fetching: boolean;
  onPick: (name: string) => void;
  onSkip: () => void;
}) {
  return (
    <View style={wc.card}>
      <View style={wc.accent} />
      {fetching ? (
        <ActivityIndicator color={COLORS.primary} style={{ marginVertical: 16 }} />
      ) : (
        <>
          <Text style={wc.question}>Which exercise do you want to swap out?</Text>
          <View style={wc.optionsGrid}>
            {exercises.map((name) => (
              <TouchableOpacity
                key={name}
                style={wc.option}
                activeOpacity={0.75}
                onPress={() => { haptic.light(); onPick(name); }}
              >
                <Text style={wc.optionTxt} numberOfLines={2}>{name}</Text>
              </TouchableOpacity>
            ))}
          </View>
          <TouchableOpacity style={{ marginTop: 14, alignSelf: "center" }} onPress={() => { haptic.select(); onSkip(); }} activeOpacity={0.7}>
            <Text style={{ color: COLORS.textMuted, fontFamily: FONTS.semibold, fontSize: 13 }}>Let AI decide</Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

/**
 * Replacement options for a specific exercise, straight from the library —
 * each with its demo clip so the user can SEE the movement before committing,
 * instead of accepting a name they may not recognise.
 */
function SwapAlternativesCard({
  original,
  options,
  fetching,
  gender,
  onPick,
  onCancel,
}: {
  original: string;
  options: Exercise[];
  fetching: boolean;
  gender: "male" | "female";
  onPick: (name: string) => void;
  onCancel: () => void;
}) {
  return (
    <View style={wc.card}>
      <View style={wc.accent} />
      {fetching ? (
        <>
          <Text style={wc.question}>Finding alternatives…</Text>
          <ActivityIndicator color={COLORS.primary} style={{ marginVertical: 12 }} />
        </>
      ) : (
        <>
          <Text style={wc.question}>Replace {original} with</Text>
          {options.map((ex) => (
            <TouchableOpacity
              key={ex.id}
              style={alt.row}
              activeOpacity={0.75}
              onPress={() => onPick(ex.name)}
              accessibilityRole="button"
              accessibilityLabel={`Replace with ${ex.name}`}
            >
              <View style={alt.demo}>
                <ExerciseDemoPreview
                  videoUrl={pickDemoVideo(ex, gender)}
                  thumbUrl={ex.demo_thumb_url}
                  fill
                />
              </View>
              <View style={{ flex: 1, minWidth: 0 }}>
                <Text style={alt.name} numberOfLines={2}>{ex.name}</Text>
                <View style={alt.metaRow}>
                  {/* The region, not just the muscle. Alternatives are RANKED by
                      subgroup rather than filtered to it — the column is only
                      ~62% populated, so filtering would empty the sheet on
                      shoulders — which means a row further down genuinely can
                      train a different part of the same muscle. Saying so is
                      the honest fix: the user picking a replacement for incline
                      work can see at a glance which options are also upper
                      chest. Falls back to the muscle group when the region is
                      unknown, because a blank pill reads as a bug. */}
                  <View style={alt.pill}>
                    <Text style={alt.pillTxt} numberOfLines={1}>
                      {prettyMuscleLabel(ex.subgroup || ex.muscle_group)}
                    </Text>
                  </View>
                  {/* Equipment only. Difficulty used to sit here too, but the
                      pill needs the room and the two together truncated to
                      "Dumbbell · ..." on a 375pt screen. Nothing is lost: both
                      callers pass an eligibility ctx, so every option in this
                      list is already inside the user's difficulty ceiling. */}
                  <Text style={alt.meta} numberOfLines={1}>{ex.equipment}</Text>
                </View>
              </View>
              <Ionicons name="chevron-forward" size={18} color={COLORS.textMuted} />
            </TouchableOpacity>
          ))}
          <TouchableOpacity
            style={{ marginTop: 14, alignSelf: "center" }}
            onPress={() => { haptic.select(); onCancel(); }}
            activeOpacity={0.7}
          >
            <Text style={{ color: COLORS.textMuted, fontFamily: FONTS.semibold, fontSize: 13 }}>
              Keep {original}
            </Text>
          </TouchableOpacity>
        </>
      )}
    </View>
  );
}

/** `upper_chest` → `Upper chest`. The library stores snake_case; users don't read it. */
function prettyMuscleLabel(raw?: string | null): string {
  const s = (raw ?? "").trim().replace(/[_-]+/g, " ");
  if (!s) return "";
  return s.charAt(0).toUpperCase() + s.slice(1).toLowerCase();
}

const alt = StyleSheet.create({
  row: {
    flexDirection: "row", alignItems: "center", gap: 12,
    backgroundColor: "rgba(255,255,255,0.04)",
    borderRadius: 16, padding: 10, marginBottom: 10,
    borderWidth: 1, borderColor: "rgba(255,255,255,0.07)",
  },
  demo: {
    width: 76, height: 60, borderRadius: 12,
    overflow: "hidden", backgroundColor: "#141416",
  },
  name: { color: "#FFF", fontFamily: FONTS.semibold, fontSize: 15, letterSpacing: -0.2 },
  metaRow: { flexDirection: "row", alignItems: "center", gap: 7, marginTop: 5 },
  pill: {
    backgroundColor: "rgba(216,255,42,0.12)",
    borderRadius: 6, paddingHorizontal: 7, paddingVertical: 2.5,
    borderWidth: 1, borderColor: "rgba(216,255,42,0.22)",
    flexShrink: 0, maxWidth: "62%",
  },
  pillTxt: { color: COLORS.primary, fontFamily: FONTS.semibold, fontSize: 10.5, letterSpacing: 0.2 },
  // shrinks first: the region matters more than the equipment on a narrow row
  meta: { color: COLORS.textMuted, fontFamily: FONTS.regular, fontSize: 12, textTransform: "capitalize", flexShrink: 1 },
});

const wc = StyleSheet.create({
  card: {
    backgroundColor: "#0D0D0F",
    borderRadius: 24, padding: 20, paddingLeft: 24,
    marginTop: 8, alignSelf: "stretch",
    borderWidth: 1, borderColor: "rgba(216,255,42,0.18)",
    overflow: "hidden",
  },
  accent: {
    position: "absolute", left: 0, top: 0, bottom: 0,
    width: 3.5, backgroundColor: COLORS.primary, borderRadius: 2,
  },
  progressRow: { flexDirection: "row", gap: 6, marginBottom: 12 },
  progressDot: { flex: 1, height: 3, borderRadius: 2, backgroundColor: "#1E1E1E" },
  progressDotActive: { backgroundColor: COLORS.primary },
  stepTxt: { color: COLORS.textMuted, fontFamily: FONTS.semibold, fontSize: 11, letterSpacing: 0.5, marginBottom: 8 },
  question: { color: "#FFF", fontFamily: FONTS.bold, fontSize: 19, letterSpacing: -0.3, marginBottom: 18, lineHeight: 25 },
  optionsGrid: { flexDirection: "row", flexWrap: "wrap", gap: 10 },
  option: {
    flexBasis: "47%", flexGrow: 1,
    backgroundColor: "#161616", borderRadius: 14,
    paddingVertical: 14, paddingHorizontal: 12,
    borderWidth: 1, borderColor: "#252525",
    alignItems: "center",
  },
  optionTxt: { color: "#E8E8E8", fontFamily: FONTS.semibold, fontSize: 14 },
  genTxt: { color: "#FFF", fontFamily: FONTS.bold, fontSize: 17, textAlign: "center", marginBottom: 6 },
  genSub: { color: COLORS.textMuted, fontFamily: FONTS.regular, fontSize: 13, textAlign: "center" },
});

function ExerciseSwapCard({
  data,
  onToday,
  onAlways,
  onNo,
}: {
  data: SwapConfirmation;
  onToday: () => void;
  onAlways: () => void;
  onNo: () => void;
}) {
  return (
    <View style={cs.confirmCard}>
      <View style={cs.confirmAccent} />

      <Text style={cs.confirmTitle}>Swap exercise?</Text>
      {data.original_exercise ? (
        <Text style={cs.confirmFood}>
          {data.original_exercise} → {data.new_exercise ?? "alternative"}
        </Text>
      ) : null}

      {/* Two action buttons side by side */}
      <View style={cs.swapBtnRow}>
        <TouchableOpacity style={cs.swapSessionBtn} onPress={() => { haptic.light(); onToday(); }} activeOpacity={0.8}>
          <Text style={cs.swapSessionBtnTxt}>Just Today</Text>
        </TouchableOpacity>
        <TouchableOpacity style={cs.yesBtn} onPress={() => { haptic.light(); onAlways(); }} activeOpacity={0.8}>
          <Text style={cs.yesBtnTxt}>Always</Text>
        </TouchableOpacity>
      </View>

      {/* Cancel, text only */}
      <TouchableOpacity style={cs.cancelTextBtn} onPress={() => { haptic.select(); onNo(); }} activeOpacity={0.7}>
        <Text style={cs.cancelTextBtnTxt}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
}

function DietPreferenceCard({
  data,
  onJustToday,
  onPermanently,
  onCancel,
}: {
  data: { food_item: string };
  onJustToday: () => void;
  onPermanently: () => void;
  onCancel: () => void;
}) {
  return (
    <View style={cs.confirmCard}>
      <View style={cs.confirmAccent} />
      <Text style={cs.confirmTitle}>Never eat this again?</Text>
      <Text style={cs.confirmFood}>{data.food_item}</Text>
      <View style={cs.swapBtnRow}>
        <TouchableOpacity style={cs.swapSessionBtn} onPress={() => { haptic.light(); onJustToday(); }} activeOpacity={0.8}>
          <Text style={cs.swapSessionBtnTxt}>Just Today</Text>
        </TouchableOpacity>
        <TouchableOpacity style={cs.yesBtn} onPress={() => { haptic.medium(); onPermanently(); }} activeOpacity={0.8}>
          <Text style={cs.yesBtnTxt}>Remove Permanently</Text>
        </TouchableOpacity>
      </View>
      <TouchableOpacity style={cs.cancelTextBtn} onPress={() => { haptic.select(); onCancel(); }} activeOpacity={0.7}>
        <Text style={cs.cancelTextBtnTxt}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
}

function WeightLogCard({
  weight_kg,
  onConfirm,
  onCancel,
}: {
  weight_kg: number;
  onConfirm: (kg: number) => void;
  onCancel: () => void;
}) {
  const [kg, setKg] = useState(weight_kg > 0 ? String(weight_kg) : "");
  const [saving, setSaving] = useState(false);
  const time = useMemo(
    () => new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" }),
    []
  );
  const handleConfirm = () => {
    const val = parseFloat(kg);
    if (!val || val < 20 || val > 500) return;
    setSaving(true);
    onConfirm(val);
  };
  return (
    <View style={cs.weightCard}>
      <View style={cs.weightHeader}>
        <View style={cs.weightIconWrap}>
          <Ionicons name="scale-outline" size={18} color={COLORS.primary} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={cs.weightMealLabel}>BODY WEIGHT</Text>
          <View style={cs.weightValRow}>
            <TextInput
              style={cs.weightInput}
              value={kg}
              onChangeText={(t) => setKg(t.replace(/[^0-9.]/g, ""))}
              keyboardType="decimal-pad"
              placeholder="0.0"
              placeholderTextColor="#333"
              selectTextOnFocus
              autoFocus={weight_kg === 0}
            />
            <Text style={cs.weightUnit}>kg</Text>
          </View>
        </View>
        <Text style={cs.weightTime}>{time}</Text>
      </View>
      <PressableScale
        haptics="none" style={cs.confirmBtn} onPress={() => { haptic.light(); handleConfirm(); }} disabled={saving}>
        {saving ? (
          <ActivityIndicator color="#000" size="small" />
        ) : (
          <Text style={cs.confirmBtnTxt}>Log Weight</Text>
        )}
      </PressableScale>
      <TouchableOpacity style={cs.cancelTextBtn} onPress={() => { haptic.select(); onCancel(); }} disabled={saving} activeOpacity={0.7}>
        <Text style={cs.cancelTextBtnTxt}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
}

// ─── Water Log Card ────────────────────────────────────────────────────────────

function WaterLogCard({
  glasses,
  onConfirm,
  onCancel,
}: {
  glasses: number;
  onConfirm: (glasses: number) => void;
  onCancel: () => void;
}) {
  const [count, setCount] = useState(glasses);
  const [saving, setSaving] = useState(false);
  return (
    <View style={cs.weightCard}>
      <View style={cs.weightHeader}>
        <View style={cs.weightIconWrap}>
          <Ionicons name="water-outline" size={18} color={COLORS.blue} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={cs.weightMealLabel}>WATER</Text>
          <View style={[cs.weightValRow, { alignItems: "center", gap: 12 }]}>
            <TouchableOpacity
              onPress={() => { haptic.select(); setCount((c) => Math.max(1, c - 1)); }}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              accessibilityLabel="Fewer glasses" accessibilityRole="button"
            >
              <Ionicons name="remove-circle-outline" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
            <Text style={cs.weightInput}>{count}</Text>
            <TouchableOpacity
              onPress={() => { haptic.select(); setCount((c) => Math.min(20, c + 1)); }}
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}
              accessibilityLabel="More glasses" accessibilityRole="button"
            >
              <Ionicons name="add-circle-outline" size={24} color={COLORS.textSecondary} />
            </TouchableOpacity>
            <Text style={cs.weightUnit}>{count === 1 ? "glass" : "glasses"}</Text>
          </View>
        </View>
      </View>
      <PressableScale
        haptics="none"
        style={cs.confirmBtn}
        onPress={() => { haptic.light(); setSaving(true); onConfirm(count); }}
        disabled={saving}
      >
        {saving ? <ActivityIndicator color="#000" size="small" /> : <Text style={cs.confirmBtnTxt}>Log Water</Text>}
      </PressableScale>
      <TouchableOpacity style={cs.cancelTextBtn} onPress={() => { haptic.select(); onCancel(); }} disabled={saving} activeOpacity={0.7}>
        <Text style={cs.cancelTextBtnTxt}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
}

// ─── Correction Card (remove / fix a logged food entry) ──────────────────────

function CorrectionCard({
  data,
  onConfirm,
  onCancel,
}: {
  data: CorrectionConfirmation;
  onConfirm: (data: CorrectionConfirmation) => void;
  onCancel: () => void;
}) {
  const [saving, setSaving] = useState(false);
  const isCorrect = data.mode === "correct";
  const fmt = (it: { food_name: string; quantity_g?: number | null; calories?: number | null }) => {
    const parts = [
      it.quantity_g != null ? `${Math.round(it.quantity_g)}g` : null,
      it.calories != null ? `${Math.round(it.calories)} kcal` : null,
    ].filter(Boolean);
    return parts.length ? `${it.food_name} · ${parts.join(" · ")}` : it.food_name;
  };
  return (
    <View style={cs.weightCard}>
      <View style={cs.weightHeader}>
        <View style={cs.weightIconWrap}>
          <Ionicons
            name={isCorrect ? "create-outline" : "trash-outline"}
            size={18}
            color={COLORS.blue}
          />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={cs.weightMealLabel}>{isCorrect ? "FIX ENTRY" : "REMOVE ENTRY"}</Text>
          {data.items.map((it) => (
            <Text
              key={it.id}
              style={{
                fontFamily: FONTS.medium,
                fontSize: 14,
                color: COLORS.text,
                marginTop: 4,
                textDecorationLine: isCorrect ? "line-through" : "none",
                opacity: isCorrect ? 0.6 : 1,
              }}
            >
              {fmt(it)}
            </Text>
          ))}
          {isCorrect && data.corrected && (
            <Text
              style={{
                fontFamily: FONTS.semibold,
                fontSize: 14,
                color: COLORS.primary,
                marginTop: 4,
              }}
            >
              → {fmt(data.corrected)}
            </Text>
          )}
        </View>
      </View>
      <PressableScale
        haptics="none"
        style={cs.confirmBtn}
        onPress={() => { haptic.light(); setSaving(true); onConfirm(data); }}
        disabled={saving}
      >
        {saving ? (
          <ActivityIndicator color="#000" size="small" />
        ) : (
          <Text style={cs.confirmBtnTxt}>{isCorrect ? "Apply Fix" : "Remove"}</Text>
        )}
      </PressableScale>
      <TouchableOpacity style={cs.cancelTextBtn} onPress={() => { haptic.select(); onCancel(); }} disabled={saving} activeOpacity={0.7}>
        <Text style={cs.cancelTextBtnTxt}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
}

// ─── Workout Log Card (from natural-language chat logging) ────────────────────

function WorkoutLogCard({
  data,
  onConfirm,
  onCancel,
}: {
  data: WorkoutLogConfirmation;
  onConfirm: (exercises: WorkoutLogExercise[]) => void;
  onCancel: () => void;
}) {
  const [exercises, setExercises] = useState<WorkoutLogExercise[]>(data.exercises);
  const [saving, setSaving] = useState(false);

  const updateSet = (exIdx: number, setIdx: number, field: "weight_kg" | "reps", raw: string) => {
    setExercises((prev) => {
      const copy = [...prev];
      const sets = [...copy[exIdx].sets];
      if (field === "weight_kg") {
        const n = parseFloat(raw);
        sets[setIdx] = { ...sets[setIdx], weight_kg: raw === "" ? null : Number.isFinite(n) ? n : sets[setIdx].weight_kg };
      } else {
        const n = parseInt(raw, 10);
        sets[setIdx] = { ...sets[setIdx], reps: Number.isFinite(n) ? n : 0 };
      }
      copy[exIdx] = { ...copy[exIdx], sets };
      return copy;
    });
  };

  const handleConfirm = () => {
    setSaving(true);
    onConfirm(exercises);
  };

  return (
    <View style={cs.foodCard}>
      <View style={cs.foodHeader}>
        <View style={cs.foodIcon}>
          <Ionicons name="barbell" size={17} color={COLORS.primary} />
        </View>
        <View style={{ flex: 1 }}>
          <Text style={cs.mealLabel}>WORKOUT LOG</Text>
          <Text style={cs.foodName}>
            {exercises.length} exercise{exercises.length !== 1 ? "s" : ""} · tap any field to edit
          </Text>
        </View>
      </View>

      {exercises.map((ex, exIdx) => (
        <View key={`${ex.exercise}-${exIdx}`} style={wl.exBlock}>
          <Text style={wl.exName}>{ex.exercise}</Text>
          {ex.sets.map((s, setIdx) => (
            <View key={setIdx} style={wl.setRow}>
              <Text style={wl.setNum}>{setIdx + 1}</Text>
              <View style={wl.setField}>
                <TextInput
                  style={wl.setInput}
                  value={s.weight_kg != null ? String(s.weight_kg) : ""}
                  onChangeText={(t) => updateSet(exIdx, setIdx, "weight_kg", t.replace(/[^0-9.]/g, ""))}
                  keyboardType="decimal-pad"
                  placeholder="0"
                  placeholderTextColor="#3A3A3A"
                  selectTextOnFocus
                />
                <Text style={wl.setUnit}>kg</Text>
              </View>
              <Text style={wl.setX}>×</Text>
              <View style={wl.setField}>
                <TextInput
                  style={wl.setInput}
                  value={String(s.reps)}
                  onChangeText={(t) => updateSet(exIdx, setIdx, "reps", t.replace(/[^0-9]/g, ""))}
                  keyboardType="number-pad"
                  placeholder="0"
                  placeholderTextColor="#3A3A3A"
                  selectTextOnFocus
                />
                <Text style={wl.setUnit}>reps</Text>
              </View>
            </View>
          ))}
        </View>
      ))}

      <PressableScale
        haptics="none" style={cs.confirmBtn} onPress={() => { haptic.light(); handleConfirm(); }} disabled={saving}>
        {saving ? (
          <ActivityIndicator color="#000" size="small" />
        ) : (
          <Text style={cs.confirmBtnTxt}>Log Workout</Text>
        )}
      </PressableScale>
      <TouchableOpacity style={cs.cancelTextBtn} onPress={() => { haptic.select(); onCancel(); }} disabled={saving} activeOpacity={0.7}>
        <Text style={cs.cancelTextBtnTxt}>Cancel</Text>
      </TouchableOpacity>
    </View>
  );
}

const wl = StyleSheet.create({
  exBlock: { marginBottom: 14 },
  exName: {
    color: "#FFF",
    fontFamily: FONTS.bold,
    fontSize: 14.5,
    marginBottom: 8,
  },
  setRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 10,
    marginBottom: 6,
  },
  setNum: {
    width: 18,
    color: COLORS.textMuted,
    fontFamily: FONTS.bold,
    fontSize: 12,
  },
  setField: {
    flexDirection: "row",
    alignItems: "baseline",
    backgroundColor: "#161616",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
  },
  setInput: {
    color: "#FFF",
    fontFamily: FONTS.extrabold,
    fontSize: 15,
    padding: 0,
    minWidth: 26,
  },
  setUnit: {
    color: COLORS.textSecondary,
    fontFamily: FONTS.medium,
    fontSize: 11,
    marginLeft: 3,
  },
  setX: { color: COLORS.textMuted, fontFamily: FONTS.bold, fontSize: 13 },
});

// ─── Chat Exercise Card ───────────────────────────────────────────────────────

const DIFF_COLOR: Record<string, string> = {
  beginner:     "#4CAF50",
  intermediate: "#FF8A00",
  advanced:     "#FF4444",
};

// Matches the ExerciseCard design in muscles.tsx, 86px thumbnail, name,
// primary muscles, and compound/difficulty badges. Tapping opens the shared
// ExerciseDetailModal (same as muscles.tsx).
function ChatExerciseCard({
  exercise,
  onPress,
}: {
  exercise: Exercise;
  onPress: () => void;
}) {
  const diff      = exercise.difficulty?.toLowerCase() ?? "";
  const diffColor = DIFF_COLOR[diff] ?? "#777";
  const thumb = listThumbFor(exercise);

  return (
    <TouchableOpacity style={ec.exCard} onPress={() => { haptic.select(); onPress(); }} activeOpacity={0.8}>
      <View style={ec.exImgWrap}>
        {thumb ? (
          <Image source={mediaSource(thumb)} style={ec.exImg} contentFit="cover" transition={200} />
        ) : (
          <View style={ec.exImgFallback}>
            <Ionicons name="barbell" size={22} color={COLORS.primary} />
          </View>
        )}
      </View>
      <View style={ec.exCardContent}>
        <Text style={ec.exCardName} numberOfLines={1}>{exercise.name}</Text>
        {(exercise.primary_muscles?.length ?? 0) > 0 && (
          <Text style={ec.exCardMuscles} numberOfLines={1}>
            {exercise.primary_muscles.slice(0, 2).map((m) => m.replace(/_/g, " ")).join(", ")}
          </Text>
        )}
        <View style={ec.exCardBadges}>
          <View style={ec.badgeCompound}>
            <Text style={ec.badgeCompoundTxt}>{exercise.is_compound ? "Compound" : "Isolation"}</Text>
          </View>
          {!!exercise.difficulty && (
            <View style={[ec.badgeDiff, { borderColor: diffColor + "66", backgroundColor: diffColor + "15" }]}>
              <Text style={[ec.badgeDiffTxt, { color: diffColor }]}>{exercise.difficulty}</Text>
            </View>
          )}
        </View>
      </View>
    </TouchableOpacity>
  );
}

const ec = StyleSheet.create({
  exCard: {
    flexDirection: "row", alignItems: "center",
    backgroundColor: "#111", borderRadius: 14,
    borderWidth: 1, borderColor: "#1E1E1E",
    overflow: "hidden", marginTop: 6,
  },
  exImgWrap:        { width: 86, height: 86, backgroundColor: "#0D0D0D", alignItems: "center", justifyContent: "center", flexShrink: 0 },
  exImg:            { width: 86, height: 86 },
  exImgFallback:    { width: 86, height: 86, alignItems: "center", justifyContent: "center" },
  exCardContent:    { flex: 1, paddingHorizontal: 13, paddingVertical: 12 },
  exCardName:       { color: "#FFF", fontFamily: FONTS.semibold, fontSize: 14, marginBottom: 3 },
  exCardMuscles:    { color: "#555", fontFamily: FONTS.regular, fontSize: 11, marginBottom: 7, textTransform: "capitalize" },
  exCardBadges:     { flexDirection: "row", gap: 6, alignItems: "center" },
  badgeCompound:    { backgroundColor: "#1A1A1A", borderWidth: 1, borderColor: "#2A2A2A", borderRadius: 7, paddingHorizontal: 7, paddingVertical: 3 },
  badgeCompoundTxt: { color: "#666", fontFamily: FONTS.medium, fontSize: 9 },
  badgeDiff:        { borderWidth: 1, borderRadius: 7, paddingHorizontal: 7, paddingVertical: 3 },
  badgeDiffTxt:     { fontFamily: FONTS.medium, fontSize: 9 },
});

const cs = StyleSheet.create({
  confirmCard: {
    backgroundColor: "#0D0D0D",
    borderRadius: 20,
    padding: 18,
    paddingLeft: 22,
    marginTop: 8,
    maxWidth: "90%",
    alignSelf: "flex-start",
    overflow: "hidden",
  },
  confirmAccent: {
    position: "absolute",
    left: 0,
    top: 0,
    bottom: 0,
    width: 3,
    backgroundColor: COLORS.primary,
    borderRadius: 1.5,
  },
  confirmTitle: {
    color: COLORS.primary,
    fontFamily: FONTS.bold,
    fontSize: 11,
    letterSpacing: 0.5,
    textTransform: "uppercase",
    marginBottom: 6,
  },
  confirmFood: {
    color: "#FFF",
    fontFamily: FONTS.bold,
    fontSize: 17,
    marginBottom: 4,
  },
  confirmQty: {
    color: "#555",
    fontFamily: FONTS.regular,
    fontSize: 13,
    marginBottom: 10,
  },
  // ── Editable food-log card (matches the meal-card design) ──
  foodCard: {
    backgroundColor: "#0D0D0D",
    borderRadius: 22,
    padding: 18,
    marginTop: 8,
    alignSelf: "stretch",
    borderWidth: 1,
    borderColor: "rgba(216,255,42,0.22)",
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 26,
    elevation: 8,
  },
  foodHeader: { flexDirection: "row", alignItems: "center", marginBottom: 16 },
  foodIcon: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: "rgba(216,255,42,0.12)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  mealLabel: {
    color: COLORS.textMuted,
    fontFamily: FONTS.bold,
    fontSize: 10.5,
    letterSpacing: 1,
  },
  foodNameRow: { flexDirection: "row", alignItems: "center", marginTop: 3 },
  foodName: {
    color: "#FFF",
    fontFamily: FONTS.bold,
    fontSize: 16,
    flexShrink: 1,
  },
  foodSep: { color: COLORS.textMuted, fontFamily: FONTS.bold, fontSize: 16 },
  qtyInput: {
    color: "#FFF",
    fontFamily: FONTS.extrabold,
    fontSize: 16,
    padding: 0,
    minWidth: 30,
    textAlign: "left",
  },
  qtyUnit: { color: "#FFF", fontFamily: FONTS.extrabold, fontSize: 16, marginLeft: 1 },
  foodTime: { color: COLORS.textMuted, fontFamily: FONTS.regular, fontSize: 12.5, marginLeft: 8 },

  foodItemList: {
    backgroundColor: "#161616",
    borderRadius: 14,
    marginBottom: 14,
    overflow: "hidden",
  },
  foodItemRow: {
    paddingHorizontal: 14,
    paddingVertical: 10,
  },
  foodItemRowDivider: {
    borderBottomWidth: 1,
    borderBottomColor: "rgba(255,255,255,0.06)",
  },
  foodItemName: { color: "#e0e0e0", fontFamily: FONTS.semibold, fontSize: 14 },
  foodItemMacro: { color: COLORS.textMuted, fontFamily: FONTS.regular, fontSize: 12, marginTop: 2 },

  foodItemBlock: { marginBottom: 4 },
  foodItemBlockDivider: {
    borderTopWidth: 1,
    borderTopColor: "rgba(255,255,255,0.06)",
    marginTop: 10,
    paddingTop: 10,
  },
  foodItemBlockName: {
    color: COLORS.textMuted,
    fontFamily: FONTS.semibold,
    fontSize: 12,
    letterSpacing: 0.6,
    textTransform: "uppercase",
    marginBottom: 10,
  },

  quantityRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "#161616",
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.05)",
  },
  quantityLabel: { color: COLORS.textMuted, fontFamily: FONTS.bold, fontSize: 12, flex: 1 },
  quantityInput: {
    color: "#FFF",
    fontFamily: FONTS.extrabold,
    fontSize: 20,
    padding: 0,
    minWidth: 40,
    textAlign: "right",
  },
  quantityUnit: { color: COLORS.textSecondary, fontFamily: FONTS.bold, fontSize: 13 },

  totalsRow: {
    flexDirection: "row",
    alignItems: "center",
    gap: 6,
    backgroundColor: "#161616",
    borderRadius: 12,
    paddingHorizontal: 14,
    paddingVertical: 10,
    marginBottom: 14,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.05)",
  },
  totalsLabel: { color: COLORS.textMuted, fontFamily: FONTS.bold, fontSize: 11, letterSpacing: 0.8, flex: 1 },
  totalsValue: { color: "#e0e0e0", fontFamily: FONTS.semibold, fontSize: 13 },
  totalsDot: { color: "#444", fontSize: 12 },

  macroGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 10,
    marginBottom: 14,
  },
  macroCell: {
    backgroundColor: "#161616",
    borderRadius: 14,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.05)",
    flexGrow: 1,
    flexBasis: "46%",
  },
  macroLbl: { color: COLORS.textMuted, fontFamily: FONTS.bold, fontSize: 10.5, letterSpacing: 0.8 },
  macroValRow: { flexDirection: "row", alignItems: "flex-end", marginTop: 6 },
  macroInput: {
    color: "#FFF",
    fontFamily: FONTS.extrabold,
    fontSize: 23,
    padding: 0,
    minWidth: 40,
    maxWidth: 96,
  },
  macroUnit: {
    color: COLORS.textSecondary,
    fontFamily: FONTS.bold,
    fontSize: 14,
    marginLeft: 2,
    marginBottom: 3,
  },

  statusPill: {
    flexDirection: "row",
    alignItems: "center",
    alignSelf: "flex-start",
    gap: 6,
    backgroundColor: "rgba(216,255,42,0.10)",
    borderRadius: 40,
    paddingHorizontal: 12,
    paddingVertical: 7,
    marginBottom: 16,
  },
  statusPillTxt: { color: COLORS.primary, fontFamily: FONTS.bold, fontSize: 12 },

  confirmBtn: {
    width: "100%",
    height: 52,
    borderRadius: 14,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 2,
  },
  confirmBtnTxt: { color: "#000", fontFamily: FONTS.bold, fontSize: 15 },

  yesBtn: {
    width: "100%",
    height: 52,
    borderRadius: 14,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 4,
  },
  yesBtnTxt: { color: "#000", fontFamily: FONTS.bold, fontSize: 15 },

  cancelTextBtn: {
    width: "100%",
    paddingVertical: 10,
    alignItems: "center",
  },
  cancelTextBtnTxt: { color: "#555", fontFamily: FONTS.medium, fontSize: 14 },

  // ── Weight log card ──
  weightCard: {
    backgroundColor: "#0D0D0D",
    borderRadius: 22,
    padding: 18,
    marginTop: 8,
    alignSelf: "stretch",
    borderWidth: 1,
    borderColor: "rgba(216,255,42,0.22)",
    shadowColor: COLORS.primary,
    shadowOffset: { width: 0, height: 8 },
    shadowOpacity: 0.18,
    shadowRadius: 26,
    elevation: 8,
  },
  weightHeader: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 18,
  },
  weightIconWrap: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: "rgba(216,255,42,0.12)",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 12,
  },
  weightMealLabel: {
    color: COLORS.textMuted,
    fontFamily: FONTS.bold,
    fontSize: 10.5,
    letterSpacing: 1,
  },
  weightValRow: { flexDirection: "row", alignItems: "flex-end", marginTop: 4 },
  weightInput: {
    color: COLORS.primary,
    fontFamily: FONTS.extrabold,
    fontSize: 36, letterSpacing: -0.8,
    padding: 0,
    minWidth: 60,
  },
  weightUnit: {
    color: COLORS.primary,
    fontFamily: FONTS.bold,
    fontSize: 18,
    marginLeft: 4,
    marginBottom: 5,
  },
  weightTime: {
    color: COLORS.textMuted,
    fontFamily: FONTS.regular,
    fontSize: 12.5,
    marginLeft: 8,
    alignSelf: "flex-start",
  },

  swapBtnRow: { flexDirection: "row", gap: 10, marginBottom: 4 },
  swapSessionBtn: {
    flex: 1,
    height: 52,
    borderRadius: 14,
    borderWidth: 1,
    borderColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
  },
  swapSessionBtnTxt: { color: COLORS.primary, fontFamily: FONTS.bold, fontSize: 15 },
});

// ─── Animated typing indicator ────────────────────────────────────────────────
// Three dots that pulse + bob in a staggered loop while the coach is "thinking".
function TypingDots() {
  const d1 = useSharedValue(0);
  const d2 = useSharedValue(0);
  const d3 = useSharedValue(0);

  useEffect(() => {
    const loop = () =>
      withRepeat(
        withSequence(
          // 520ms, not 360. The coach can take tens of seconds to answer, and a
          // fast pulse reads as impatience rather than thought.
          withTiming(1, { duration: 520, easing: Easing.inOut(Easing.quad) }),
          withTiming(0, { duration: 520, easing: Easing.inOut(Easing.quad) })
        ),
        -1,
        false
      );
    d1.value = loop();
    d2.value = withDelay(200, loop());
    d3.value = withDelay(400, loop());
  }, [d1, d2, d3]);

  const st1 = useAnimatedStyle(() => ({ opacity: 0.35 + d1.value * 0.65, transform: [{ translateY: -d1.value * 4 }] }));
  const st2 = useAnimatedStyle(() => ({ opacity: 0.35 + d2.value * 0.65, transform: [{ translateY: -d2.value * 4 }] }));
  const st3 = useAnimatedStyle(() => ({ opacity: 0.35 + d3.value * 0.65, transform: [{ translateY: -d3.value * 4 }] }));

  return (
    <View style={s.typingIndicator}>
      <Animated.View style={[s.typingDot, st1]} />
      <Animated.View style={[s.typingDot, st2]} />
      <Animated.View style={[s.typingDot, st3]} />
    </View>
  );
}

// ─── Markdown-lite text ────────────────────────────────────────────────────────
// The model replies with **bold** markers; render them instead of showing raw
// asterisks. Only bold is supported, anything heavier belongs in a card.
function MdText({ text, style }: { text: string; style: any }) {
  const parts = String(text ?? "").split(/(\*\*[^*]+\*\*)/g);
  return (
    <Text style={style}>
      {parts.map((p, i) =>
        p.startsWith("**") && p.endsWith("**") ? (
          <Text key={i} style={{ fontFamily: FONTS.bold }}>{p.slice(2, -2)}</Text>
        ) : (
          p
        )
      )}
    </Text>
  );
}

// ─── Typewriter reveal ─────────────────────────────────────────────────────────
// Reveals a fresh reply word-by-word (ChatGPT style). Only used for messages
// flagged animate, history renders instantly. onTick lets the parent keep the
// scroll pinned to the growing bubble; onDone marks the message as revealed.
function TypewriterMdText({
  text, style, onTick, onDone,
}: {
  text: string; style: any; onTick?: () => void; onDone?: () => void;
}) {
  // Split keeping whitespace tokens so join reconstructs the exact text
  const words = React.useMemo(() => String(text ?? "").split(/(\s+)/), [text]);
  const [visible, setVisible] = useState(0);
  const doneRef = useRef(false);

  useEffect(() => {
    let i = 0;
    // Reveal scales with length so a long plan finishes in ~1s instead of
    // crawling word-by-word for 7s. Short replies still reveal ~word-by-word.
    // step is kept even so each tick advances whole word+whitespace pairs.
    const step = Math.max(1, Math.ceil(words.length / 2 / 45)) * 2;
    const timer = setInterval(() => {
      i += step;
      if (i >= words.length) {
        clearInterval(timer);
        setVisible(words.length);
        if (!doneRef.current) { doneRef.current = true; onDone?.(); }
      } else {
        setVisible(i);
        if ((i / 2) % 4 === 0) onTick?.();
      }
    }, 22);
    return () => clearInterval(timer);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [text]);

  // Hide a trailing unpaired ** so bold markers never flash as raw asterisks
  let shown = words.slice(0, visible).join("");
  if (((shown.match(/\*\*/g)?.length ?? 0) % 2) === 1) {
    shown = shown.replace(/\*\*(?!.*\*\*)/, "");
  }
  return <MdText style={style} text={shown} />;
}

// ─── Chat backdrop ─────────────────────────────────────────────────────────────
// Faint branded watermark painted behind the message list. Visible as an empty-
// state hero when the freshest view has no content; quietly sits behind bubbles
// once a conversation fills the screen. Non-interactive.
function ChatBackdrop({ coach }: { coach: any }) {
  return (
    <View style={s.backdrop} pointerEvents="none">
      <View style={s.backdropRing}>
        {coach?.profile_image ? (
          <Image source={{ uri: coach.profile_image }} style={s.backdropAvatar} />
        ) : (
          <Ionicons name="sparkles" size={46} color="rgba(216,255,42,0.18)" />
        )}
      </View>
      <Text style={s.backdropKicker}>AI COACH</Text>
      <Text style={s.backdropName}>{coach?.name ?? "Your Coach"}</Text>
      <Text style={s.backdropTagline}>Training · Nutrition · Recovery</Text>
    </View>
  );
}

// ─── Screen ───────────────────────────────────────────────────────────────────

export default function CoachScreen() {
  const { user } = useAuth();
  const { isCreator, creatorProfile } = useCreator();
  const params = useLocalSearchParams<{ prefill?: string }>();

  const [coaches, setCoaches] = useState<any[]>([]);
  const [activeCoach, setActiveCoach] = useState<any>(null);
  // Coach picker for users with no coach yet, up to 4 suggested creators.
  const [suggestedCreators, setSuggestedCreators] = useState<any[]>([]);
  const [confirmCreator, setConfirmCreator] = useState<any>(null);
  const [subscribing, setSubscribing] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [inputFocused, setInputFocused] = useState(false);

  // Draft persistence. The coach tab unmounts when you navigate away, so a
  // half-typed message was simply gone on return — worst exactly when it costs
  // most: you type something long, get bored waiting for the reply, check
  // another tab, and lose it.
  const draftRestored = useRef(false);
  useEffect(() => {
    storage.get(COACH_DRAFT_KEY)
      .then((saved) => {
        // Only fills an EMPTY box, so a restored draft can never clobber
        // something the user started typing while the read was in flight.
        // eslint-disable-next-line react-hooks/set-state-in-effect
        if (saved) setInput((cur) => cur || saved);
      })
      .catch(() => {})
      .finally(() => { draftRestored.current = true; });
  }, []);

  useEffect(() => {
    // Nothing may be written until the restore finishes. `input` starts empty,
    // so without this the save below fires its "empty means delete" branch and
    // wipes the stored draft out from under the read still in flight.
    if (!draftRestored.current) return;
    const t = setTimeout(() => {
      (input.trim()
        ? storage.set(COACH_DRAFT_KEY, input)
        : storage.remove(COACH_DRAFT_KEY)
      ).catch(() => {});
    }, 400);
    return () => clearTimeout(t);
  }, [input]);
  // The tab bar is absolutely positioned (see (tabs)/_layout), so this pinned
  // composer has to clear it itself.
  const tabBarHeight = useTabBarHeight();
  const keyboardUp = useKeyboardVisible();
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [loadingMsgIdx, setLoadingMsgIdx] = useState(0);
  const loadingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const [coachPickerOpen, setCoachPickerOpen] = useState(false);
  // Measured, not a constant: the header is an overlay so that messages pass
  // UNDER its material (a blur with nothing behind it is just an expensive
  // flat fill). The scroll content has to reserve exactly its height, and that
  // height moves with the safe-area inset and Dynamic Type, so it is read off
  // the real layout rather than guessed.
  const [headerH, setHeaderH] = useState(0);
  const [profile, setProfile] = useState<any>(null);
  const [preferences, setPreferences] = useState<any>(null);
  const [messagesRemaining, setMessagesRemaining] = useState<number | null>(null);
  const [upgradeRequired, setUpgradeRequired] = useState(false);

  const [upgradeMessage, setUpgradeMessage] = useState("You've reached your daily message limit.");
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);
  const [planBrief, setPlanBrief] = useState<PlanBriefState>(null);
  const [briefGender, setBriefGender] = useState<"male" | "female">("male");
  const [adjustWizard, setAdjustWizard] = useState<WizardState>(null);
  const [cheatWizard, setCheatWizard] = useState<WizardState>(null);
  const [swapWizard, setSwapWizard] = useState<WizardState>(null);
  // DB-grounded replacement options for the picked exercise. When this is set we
  // skip the LLM entirely: the user chose the exercise AND chooses the
  // replacement, so the swap is instant, free, and can't be hallucinated.
  const [swapAlts, setSwapAlts] = useState<
    { original: string; options: Exercise[]; fetching: boolean } | null
  >(null);
  const [swapPicker, setSwapPicker] = useState<{ exercises: string[]; fetching: boolean } | null>(null);
  const [swapTargetExercise, setSwapTargetExercise] = useState<string | null>(null);
  const [liveContext, setLiveContext] = useState<LiveUserContext | null>(null);
  const [selectedExercise, setSelectedExercise] = useState<Exercise | null>(null);
  const [foodCardReady, setFoodCardReady] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const foodCardTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Most recent tap-to-act suggestion (meals or exercises). A plain "yes" /
  // "add it" reply right after one of these executes it locally, instead of
  // round-tripping to /chat just to have the model say "sure, adding now".
  const pendingSuggestionRef = useRef<
    | { kind: "meals"; data: SuggestedMeal[] }
    | { kind: "exercises"; data: SuggestedExercise[] }
    | null
  >(null);

  // Ids of the most recently confirmed coach-memory fact(s). A bare "forget
  // that" reply right after archives them locally without a round trip.
  const lastConfirmedFactIdsRef = useRef<string[] | null>(null);

  // Pagination: load 10 messages at a time, fetch older batches on scroll-up
  const [hasMore, setHasMore] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const oldestCursorRef = useRef<string | null>(null); // created_at of oldest loaded msg
  const loadingMoreRef = useRef(false);                // guards concurrent fetches
  const lastStampRef = useRef<string>("");             // tracks bottom msg for autoscroll

  const hasLoaded = useRef(false);
  const scrollRef = useRef<ScrollView>(null);
  // Replies that already finished their typewriter reveal (never re-animate)
  const revealedIdsRef = useRef<Set<string>>(new Set());
  // When true, newly-mounted bubbles skip their entry animation. Set during
  // scroll-up prepends so older batches appear in place instead of cascading.
  const suppressEnterRef = useRef(false);
  // When true, the next content render jumps straight to the latest message
  // (no animation). Set on first load, tab refocus, and coach switch so the
  // chat always opens at the bottom rather than wherever it was last parked.
  const wantBottomRef = useRef(true);
  // Lightweight {name, slug} index for detecting exercise mentions in replies.
  const exerciseIndexRef = useRef<{ name: string; slug: string }[]>([]);

  useEffect(() => {
    if (sending) {
      setLoadingMsgIdx(0);
      loadingIntervalRef.current = setInterval(() => {
        setLoadingMsgIdx((i) => (i + 1) % COACH_LOADING_MSGS.length);
      }, 1800);
    } else {
      if (loadingIntervalRef.current) { clearInterval(loadingIntervalRef.current); loadingIntervalRef.current = null; }
    }
    return () => { if (loadingIntervalRef.current) clearInterval(loadingIntervalRef.current); };
  }, [sending]);

  // Load the exercise name index once so we can attach cards for exercises the
  // coach names in plain text (fallback when the backend omits recommendations).
  useEffect(() => {
    getExerciseLibraryLite()
      .then((list) => {
        exerciseIndexRef.current = list
          .map((e) => ({ name: e.name, slug: e.slug }))
          .filter((e) => e.name && e.slug)
          .sort((a, b) => b.name.length - a.name.length);
      })
      .catch(() => {});
  }, []);

  useFocusEffect(
    useCallback(() => {
      // Reopening the tab should always land on the latest message.
      wantBottomRef.current = true;
      loadInitial();
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [user, isCreator, creatorProfile?.id])
  );

  // ─── Voice input (on-device speech-to-text) ────────────────────────────────
  // Speech events are module-global: mic buttons on other screens (CoachFAB,
  // creator insights) fire them too, and this tab stays mounted. Only react
  // to sessions this screen started.
  const voiceOwnerRef = useRef(false);
  useSpeechRecognitionEvent("start", () => {
    if (voiceOwnerRef.current) setIsListening(true);
  });
  useSpeechRecognitionEvent("end", () => {
    if (voiceOwnerRef.current) { voiceOwnerRef.current = false; setIsListening(false); }
  });
  useSpeechRecognitionEvent("result", (event) => {
    if (!voiceOwnerRef.current) return;
    const transcript = event.results[0]?.transcript;
    if (transcript != null) setInput(transcript);
  });
  useSpeechRecognitionEvent("error", (event) => {
    if (__DEV__) console.log("[coach] speech recognition error:", event.error, event.message);
    if (voiceOwnerRef.current) { voiceOwnerRef.current = false; setIsListening(false); }
  });

  // Stop listening automatically if the tab loses focus mid-recording.
  useFocusEffect(
    useCallback(() => {
      return () => {
        ExpoSpeechRecognitionModule.stop();
      };
    }, [])
  );

  const toggleVoiceInput = async () => {
    if (isListening) {
      haptic.light();
      ExpoSpeechRecognitionModule.stop();
      return;
    }
    const permission = await ExpoSpeechRecognitionModule.requestPermissionsAsync();
    if (!permission.granted) {
      appendLocal([
        {
          role: "assistant",
          content: "I need microphone access to hear you, enable it for Rippd in your device Settings.",
        },
      ]);
      return;
    }
    haptic.light();
    setInput("");
    voiceOwnerRef.current = true;
    ExpoSpeechRecognitionModule.start({
      lang: "en-US",
      interimResults: true,
      continuous: false,
    });
  };

  const micPulse = useSharedValue(0);
  useEffect(() => {
    micPulse.value = isListening
      ? withRepeat(
          withSequence(
            withTiming(1, { duration: 550, easing: Easing.inOut(Easing.quad) }),
            withTiming(0, { duration: 550, easing: Easing.inOut(Easing.quad) })
          ),
          -1,
          false
        )
      : withTiming(0, { duration: 200 });
  }, [isListening, micPulse]);
  const micPulseStyle = useAnimatedStyle(() => ({
    transform: [{ scale: 1 + micPulse.value * 0.14 }],
  }));

  // Handle prefill param from diet plan "Log this meal"
  useEffect(() => {
    if (params.prefill && !loading) {
      setInput(decodeURIComponent(params.prefill));
    }
  }, [params.prefill, loading]);

  const loadInitial = async (force = false) => {
    if (!user) return;
    if (!hasLoaded.current) setLoading(true);

    // Wake the backend the moment this screen opens, not when the user finally
    // hits send. Render's free tier spins the dyno down after ~15 min idle and
    // the boot costs ~10s, which the user used to eat in full while staring at
    // typing dots. Opening the coach, picking a coach and typing a message
    // covers most of that. Deliberately not awaited, this must never delay the
    // screen, and a failure here just means the first message pays the boot.
    get("/ping").catch(() => {});

    try {
      // Only the coach list is critical to this screen. Profile / preferences /
      // live-context are enrichment, if one of them has a transient failure it
      // must NOT reject the whole load and leave `coaches` empty, which would
      // show "No coaches yet" to a user who actually has a coach. So each of the
      // non-critical calls degrades to null instead of taking the screen down.
      const [coachList, prof, prefs, liveCtx] = await Promise.all([
        getFollowedCoaches(user.id, force),
        getProfile(user.id, force).catch((e) => { if (__DEV__) console.log("profile load failed", e); return null; }),
        getPreferences(user.id, force).catch((e) => { if (__DEV__) console.log("prefs load failed", e); return null; }),
        fetchLiveUserContext(user.id).catch(() => null),
      ]);
      setCoaches(coachList);
      if (prof) setProfile(prof);
      if (prefs) setPreferences(prefs);
      setLiveContext(liveCtx);

      if (coachList.length > 0) {
        const first = coachList[0];
        setActiveCoach(first);
        await loadConversation(first.id);
      } else if (isCreator && creatorProfile) {
        // A creator is their own AI coach, chat runs against their own persona
        // (coaching/diet philosophy, tone). No coach-picker for them.
        const selfCoach = {
          id: creatorProfile.id,
          name: creatorProfile.name,
          profile_image: creatorProfile.profile_image,
          specialty: creatorProfile.specialty,
        };
        setCoaches([selfCoach]);
        setActiveCoach(selfCoach);
        await loadConversation(selfCoach.id);
      } else {
        // No coach yet → load up to 4 suggestions for the picker.
        getCreators()
          .then((cs) => setSuggestedCreators((cs ?? []).slice(0, 4)))
          .catch(() => {});
      }
    } catch (e) {
      if (__DEV__) console.log(e);
    } finally {
      hasLoaded.current = true;
      setLoading(false);
    }
  };

  // Subscribe to the confirmed creator, then reload so the tab becomes a normal
  // coached account (chat, header, etc.).
  const handleConfirmSelectCreator = async () => {
    if (!user || !confirmCreator || subscribing) return;
    setSubscribing(true);
    try {
      await followCreator(user.id, confirmCreator.id);
      trackAction("coach_selected", { creator_id: confirmCreator.id, from: "coach_picker" });
      haptic.success();
      setConfirmCreator(null);
      await loadInitial(true);
    } catch {
      haptic.error();
      toast.error("Couldn't select that coach, please try again.");
    } finally {
      setSubscribing(false);
    }
  };

  // Loads the most recent page (newest 10 messages within the last 7 days).
  const loadConversation = async (creatorId: string) => {
    if (!user) return;
    oldestCursorRef.current = null;
    loadingMoreRef.current = false;
    setLoadingMore(false);
    try {
      const [page, pending, pendingFood] = await Promise.all([
        getConversationPage(user.id, creatorId, null, CHAT_PAGE_SIZE),
        getPendingProposal(user.id, creatorId).catch(() => null),
        getPendingFoodLog(user.id, creatorId).catch(() => null),
      ]);
      const msgs: Message[] = page.map((m) => ({
        id: m.id,
        role: m.role,
        // A brief sends a long composed sentence because that is what the model
        // reads, while the bubble shows a short label. Only the sentence is
        // stored, so without this the thread changed under you on reload: three
        // words became a paragraph of machine-ish instructions.
        content:
          (m.meta?.card === "plan_brief" && m.meta.brief?.display) || m.content,
        created_at: m.created_at,
        // Redraw whatever card this message carried. Without this a proposal
        // read back from history is plain text, which is how cancelling one
        // nine minutes after it appeared left nothing on screen.
        ...(m.meta?.card === "program_proposal" && m.meta.args
          ? { programProposal: m.meta.args as ProgramProposalData }
          : {}),
        ...(m.meta?.card === "plan_brief" && m.meta.brief
          ? { planBrief: m.meta.brief as PlanBriefSummary }
          : {}),
      }));

      // If the backend saved a program proposal that never got confirmed (e.g.
      // the live request timed out on the client), re-render it as the proper
      // interactive card instead of the raw day-by-day confirmation text.
      const proposalArgs =
        pending?.tool === "propose_workout_program" ? pending.args : null;
      if (proposalArgs) {
        // Replace the ugly saved confirmation bubble with just the short
        // rationale, the card below carries the full breakdown.
        const isConfirmationText = (c: string) =>
          /to activate it or\s+\*\*no\*\*\s+to cancel/i.test(c) ||
          /to save this plan or\s+\*\*no\*\*\s+to cancel/i.test(c);
        for (let i = msgs.length - 1; i >= 0; i--) {
          if (msgs[i].role === "assistant" && isConfirmationText(msgs[i].content)) {
            msgs[i] = {
              ...msgs[i],
              content: proposalArgs.coaching_rationale || "Here's the program I built for you.",
            };
            break;
          }
        }
      }

      setMessages(msgs);
      oldestCursorRef.current = msgs[0]?.created_at ?? null;
      setHasMore(page.length === CHAT_PAGE_SIZE);

      // Surface the interactive proposal card so the user can activate/cancel.
      if (proposalArgs) {
        setPendingAction({
          type: "program_proposal",
          data: { ...proposalArgs, isModify: pending?.intent === "modify_program" },
        });
      } else if (pendingFood?.length) {
        // Rehydrate an unconfirmed food log (fresh within 6h) so the editable
        // macro table survives an app restart, same treatment proposals get.
        setPendingAction({
          type: "food_log",
          data: pendingFood.map((d) => ({
            food_name: d.food_name,
            count:     d.count ?? null,
            unit:      d.unit ?? null,
            quantity:  d.quantity_g ?? null,
            calories:  d.calories ?? null,
            protein_g: d.protein_g ?? null,
            carbs_g:   d.carbs_g ?? null,
            fats_g:    d.fat_g ?? null,
            meal_type: (d.meal_type as FoodConfirmation["meal_type"]) ?? null,
          })),
        });
        setFoodCardReady(true); // no "fetching macros" delay on rehydrate
      }
    } catch {
      setMessages([]);
      setHasMore(false);
    }
  };

  // Fetches the next older batch of 10 and prepends it (triggered by scroll-up).
  const loadOlder = async () => {
    if (loadingMoreRef.current || !hasMore || !user || !activeCoach) return;
    const cursor = oldestCursorRef.current;
    if (!cursor) return;
    loadingMoreRef.current = true;
    setLoadingMore(true);
    try {
      const page = await getConversationPage(user.id, activeCoach.id, cursor, CHAT_PAGE_SIZE);
      if (page.length === 0) {
        setHasMore(false);
        return;
      }
      const older: Message[] = page.map((m) => ({
        id: m.id,
        role: m.role,
        content: m.content,
        created_at: m.created_at,
      }));
      // Suppress entry animation for this prepend so the older batch doesn't
      // cascade in; reset on the next frame for subsequent new messages.
      suppressEnterRef.current = true;
      setMessages((prev) => {
        const seen = new Set(prev.map((p) => p.id));
        const fresh = older.filter((m) => !seen.has(m.id));
        return [...fresh, ...prev];
      });
      requestAnimationFrame(() => { suppressEnterRef.current = false; });
      oldestCursorRef.current = page[0].created_at;
      setHasMore(page.length === CHAT_PAGE_SIZE);
    } catch {
      // leave hasMore as-is so the user can retry by scrolling again
    } finally {
      loadingMoreRef.current = false;
      setLoadingMore(false);
    }
  };

  const onScroll = (e: { nativeEvent: { contentOffset: { y: number } } }) => {
    if (e.nativeEvent.contentOffset.y <= 200) loadOlder();
  };

  const switchCoach = async (coach: any) => {
    setActiveCoach(coach);
    setCoachPickerOpen(false);
    setMessages([]);
    setPendingAction(null);
    setHasMore(false);
    oldestCursorRef.current = null;
    lastStampRef.current = "";
    wantBottomRef.current = true;
    await loadConversation(coach.id);
  };

  // Auto-scroll to the bottom only when a NEW message lands at the bottom (or on
  // first load), never on a scroll-up prepend, which keeps the older batch in view.
  useEffect(() => {
    const last = messages[messages.length - 1];
    const stamp = last ? `${last.id}` : "";
    // Forced open-at-latest (first load / refocus / coach switch): jump to the
    // bottom instantly, no animation. Wait until messages actually exist so the
    // empty first render doesn't consume the intent.
    if (wantBottomRef.current) {
      if (messages.length > 0) {
        wantBottomRef.current = false;
        lastStampRef.current = stamp;
        setTimeout(() => scrollRef.current?.scrollToEnd({ animated: false }), 0);
      }
      return;
    }
    if (stamp !== lastStampRef.current) {
      lastStampRef.current = stamp;
      setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
    }
  }, [messages]);

  useEffect(() => {
    if (pendingAction) setTimeout(() => scrollRef.current?.scrollToEnd({ animated: true }), 100);
  }, [pendingAction]);

  // Text the user actually TYPED, as opposed to a prompt the app composed for
  // itself (the plan brief, quick prompts, confirmations). Only user text is
  // routed into the brief: doing this inside sendMessage would make the brief's
  // own generated instruction match and reopen the brief, forever.
  const submitUserText = async (raw?: string) => {
    const content = (raw ?? input).trim();
    if (!content || sending) return;
    if (wantsProgramBuild(content)) {
      setInput("");
      startProgramWizard();
      return;
    }
    await sendMessage(content);
  };

  const sendMessage = async (
    text?: string,
    // `display` shows something other than what is sent — the plan brief
    // sends a full instruction but shows the user's own words, with the
    // brief card attached. Without it the thread fills with prompt text.
    opts?: { display?: string; planBrief?: PlanBriefSummary }
  ) => {
    const content = (text ?? input).trim();
    if (!content || !activeCoach || !user || sending) return;

    // Here rather than on the send button: this is also reached by the return
    // key (onSubmitEditing) and by the trailing-newline path in onChangeText,
    // neither of which goes through the button's own press feedback.
    haptic.light();

    // Detect food dislike BEFORE sending to AI
    const detectedFood = detectFoodDislike(content);
    if (detectedFood) {
      setInput("");
      setSending(false);
      appendLocal([{ role: "user", content }], true);
      setPendingAction({ type: "diet_preference", data: { food_item: detectedFood } });
      return;
    }

    // A bare "yes"/"add it" right after a meal or exercise suggestion executes
    // it locally, no need to round-trip to the model just to say "sure, adding".
    const pendingSuggestion = pendingSuggestionRef.current;
    if (pendingSuggestion && AFFIRM_ADD_RE.test(content)) {
      pendingSuggestionRef.current = null;
      setInput("");
      appendLocal([{ role: "user", content }], true);
      try {
        if (pendingSuggestion.kind === "exercises") {
          const res = await addExercisesToPlan(pendingSuggestion.data);
          appendLocal(
            [{
              role: "assistant",
              content: "needsPlan" in res
                ? "You don't have an active program yet, want me to build one?"
                : `Added to ${res.dayTitle}. Go crush it.`,
            }],
            true
          );
        } else {
          for (const m of pendingSuggestion.data) await logSuggestedMeal(m);
          appendLocal([{ role: "assistant", content: `Logged all ${pendingSuggestion.data.length} item${pendingSuggestion.data.length === 1 ? "" : "s"}.` }], true);
        }
      } catch {
        appendLocal([{ role: "assistant", content: "Couldn't add that just now, try tapping the card above instead." }]);
      }
      return;
    }

    // A bare "forget that" right after confirming a coach-memory fact archives
    // it locally, no round trip to the model needed.
    const lastFactIds = lastConfirmedFactIdsRef.current;
    if (lastFactIds && FORGET_RE.test(content)) {
      lastConfirmedFactIdsRef.current = null;
      setInput("");
      appendLocal([{ role: "user", content }], true);
      try {
        await archiveCoachMemoryFacts(user.id, lastFactIds);
        appendLocal([{ role: "assistant", content: "Got it, forgotten." }], true);
      } catch {
        appendLocal([{ role: "assistant", content: "Couldn't forget that just now, you can also remove it from Profile > Coach Memory." }]);
      }
      return;
    }

    setInput("");
    setSending(true);
    setPendingAction(null);
    haptic.light();

    const userMsg: Message = {
      id: newLocalId(),
      role: "user",
      content: opts?.display ?? content,
      created_at: nowIso(),
      ...(opts?.planBrief ? { planBrief: opts.planBrief } : {}),
    };
    setMessages((prev) => [...prev, userMsg]);

    try {
      // Drives which follow-up card we render below, NOT the coach's context.
      // No user_context in the payload on purpose: the backend builds and caches
      // its own (ai/context_builder.py) and explicitly ignores whatever the
      // client sends, so building one here only burned queries to produce a
      // string the server threw away. Awareness fixes go in context_builder.py.
      const topic = detectTopic(content);
      const payload = {
        user_id: user.id,
        message: content,
        creator_id: activeCoach.id,
        // Stored as this message's meta so the recap card can be redrawn after a
        // reload. The model never reads it — the same choices are already in the
        // message text, and sending them twice would let the echo drift from it.
        ...(opts?.planBrief ? { brief: opts.planBrief } : {}),
      };

      // Program/diet generation is a large structured LLM call (up to 8000
      // tokens) that routinely takes well past the default 15s timeout -
      // give /chat much more room before treating it as failed.
      const CHAT_TIMEOUT_MS = 90_000;

      let response: ChatResponse;
      try {
        response = await post<ChatResponse>("/chat", payload, CHAT_TIMEOUT_MS);
      } catch (firstErr: any) {
        const isTimeout =
          firstErr?.name === "AbortError" ||
          firstErr?.message?.toLowerCase().includes("abort") ||
          firstErr?.message?.toLowerCase().includes("timeout") ||
          firstErr?.message?.toLowerCase().includes("network") ||
          firstErr?.status === 503 ||
          firstErr?.status === 504;

        if (isTimeout) {
          // Show waking up message and retry once
          const wakingMsg: Message = {
            id: newLocalId(),
            role: "assistant",
            content: "Give me a second, just waking up...",
            created_at: nowIso(),
          };
          setMessages((prev) => [...prev, wakingMsg]);
          await new Promise((r) => setTimeout(r, 4000));
          try {
            response = await post<ChatResponse>("/chat", payload, CHAT_TIMEOUT_MS);
          } finally {
            // Remove the waking up message once real response arrives (or fails)
            setMessages((prev) => prev.filter((m) => m.id !== wakingMsg.id));
          }
        } else {
          throw firstErr;
        }
      }

      if (response.messages_remaining != null) {
        setMessagesRemaining(response.messages_remaining);
      }

      if (response.upgrade_required) {
        setUpgradeRequired(true);
        if (response.upgrade_message) setUpgradeMessage(response.upgrade_message);
        const upgradeMsg: Message = {
          id: newLocalId(),
          role: "assistant",
          content: tidyMarkdown(
            response.upgrade_message ??
              "You've used your free messages for today. Upgrade to continue."
          ),
          created_at: nowIso(),
        };
        setMessages((prev) => [...prev, upgradeMsg]);
        return;
      }

      // For program proposals, show only the coaching rationale as the text bubble
      const rawReply = response.pending_tool === "propose_workout_program" && response.pending_args?.coaching_rationale
        ? response.pending_args.coaching_rationale
        : response.response ?? "I couldn't respond right now. Try again.";
      const reply = tidyMarkdown(rawReply);
      const assistantMsgId = newLocalId();

      // Detect weight-log confirmation embedded in AI reply before displaying it
      const weightLog = !response.pending_weight_data ? extractWeightLog(reply) : null;
      const displayReply = weightLog ? weightLog.cleanText : reply;

      // Inline cards: weekly review beats today-progress when both would match,
      // and stat cards never show alongside meal/exercise cards or a proposal.
      const userAskedWeekly = /\b(week|weekly|review my|last 7)\b/i.test(content);
      const userAskedToday = /\b(today|so far|left|remaining|on track)\b/i.test(content);
      const hasMeals = !!response.suggested_meals?.length;
      const hasExercises = !!response.suggested_exercises?.length;
      const hasProposal = response.pending_tool === "propose_workout_program" && !!response.pending_args;
      const hasFacts = !!response.candidate_facts?.length;
      const hasActionCard = hasMeals || hasExercises || hasProposal || hasFacts;

      const assistantMsg: Message = {
        id: assistantMsgId,
        role: "assistant",
        content: displayReply,
        created_at: nowIso(),
        animate: true,
        ...(hasProposal
          ? { programProposal: { ...response.pending_args!, isModify: response.intent === "modify_program" } }
          : {}),
        ...(hasMeals ? { meals: response.suggested_meals! } : {}),
        ...(hasExercises ? { workoutExercises: response.suggested_exercises! } : {}),
        ...(hasFacts ? { candidateFacts: response.candidate_facts! } : {}),
        ...(!hasActionCard && userAskedWeekly ? { weeklyCard: true } : {}),
        ...(!hasActionCard && !userAskedWeekly && userAskedToday && topic === "nutrition"
          ? { progressCard: true }
          : {}),
      };
      setMessages((prev) => [...prev, assistantMsg]);

      // Track the most recent tap-to-act suggestion so a plain "yes"/"add it"
      // reply can execute it without a round trip to the backend.
      if (hasMeals) pendingSuggestionRef.current = { kind: "meals", data: response.suggested_meals! };
      else if (hasExercises) pendingSuggestionRef.current = { kind: "exercises", data: response.suggested_exercises! };
      else pendingSuggestionRef.current = null;

      // Attach exercise cards: prefer the backend's recommendations, otherwise
      // fall back to scanning the reply text for exercise names in the library.
      const recommendedSlugs = response.recommended_exercises?.length
        ? response.recommended_exercises.map((e) => e.slug)
        : detectExerciseSlugs(reply, exerciseIndexRef.current);
      if (recommendedSlugs.length) {
        getExercisesByIds(recommendedSlugs)
          .then((exs) => {
            if (exs.length > 0) {
              setMessages((prev) =>
                prev.map((m) => (m.id === assistantMsgId ? { ...m, exercises: exs } : m))
              );
            }
          })
          .catch(() => {});
      }

      // Weight log card, backend field takes priority, then client-side detection
      const weightKg = response.pending_weight_data?.weight_kg ?? weightLog?.weight_kg ?? null;
      if (weightKg != null) {
        setPendingAction({ type: "weight_log", data: { weight_kg: weightKg } });
      }

      // Water log card, detected client-side from the user's own message,
      // only when no higher-priority confirmation card claimed the slot.
      const waterFromMsg = extractWaterGlasses(content);
      if (
        waterFromMsg != null &&
        weightKg == null &&
        !response.pending_food_data &&
        !response.pending_workout_data &&
        !response.pending_swap_data
      ) {
        setPendingAction({ type: "water_log", data: { glasses: waterFromMsg } });
      }

      // Show food log confirmation card (after a 2-second hint message).
      // Older backend versions send a single object instead of an array -
      // normalize so the card renders against either shape.
      const foodData = Array.isArray(response.pending_food_data)
        ? response.pending_food_data
        : response.pending_food_data
          ? [response.pending_food_data]
          : [];
      if (foodData.length) {
        setPendingAction({
          type: "food_log",
          data: foodData.map((d) => ({
            food_name: d.food_name,
            count:     d.count ?? null,
            unit:      d.unit ?? null,
            quantity:  d.quantity_g ?? null,
            calories:  d.calories ?? null,
            protein_g: d.protein_g ?? null,
            carbs_g:   d.carbs_g ?? null,
            fats_g:    d.fat_g ?? null,
            meal_type: d.meal_type ?? null,
          })),
        });
        setFoodCardReady(false);
        if (foodCardTimerRef.current) clearTimeout(foodCardTimerRef.current);
        foodCardTimerRef.current = setTimeout(() => setFoodCardReady(true), 2000);
      }

      // Show workout-set logging card
      if (response.pending_workout_data?.exercises?.length) {
        setPendingAction({ type: "workout_log", data: response.pending_workout_data });
      }

      // Show program proposal card
      if (response.pending_tool === "propose_workout_program" && response.pending_args) {
        setPendingAction({
          type: "program_proposal",
          data: { ...response.pending_args, isModify: response.intent === "modify_program" },
        });
      }

      // Show exercise swap confirmation card
      if (response.pending_swap_data) {
        setPendingAction({
          type: "exercise_swap",
          data: {
            original_exercise: response.pending_swap_data.original_exercise,
            new_exercise:      response.pending_swap_data.replacement_exercise,
          },
        });
      }

      // Show correction card (remove / fix a previously logged food entry)
      if (response.pending_correction_data?.items?.length) {
        setPendingAction({ type: "correction", data: response.pending_correction_data });
      }
    } catch {
      // Program/diet generation is slow, the backend often finishes and saves
      // the proposal even when the client request times out. Before surfacing an
      // error, check whether the proposal actually landed server-side; if it did,
      // reload the conversation so the proper proposal card appears in place.
      let recovered = false;
      try {
        const pending = await getPendingProposal(user.id, activeCoach.id);
        if (pending?.tool === "propose_workout_program" || pending?.tool === "propose_diet_plan") {
          await loadConversation(activeCoach.id);
          recovered = true;
        }
      } catch {
        /* fall through to the generic error */
      }
      if (!recovered) {
        appendLocal([{ role: "assistant", content: "Something went wrong. Please try again." }]);
      }
    } finally {
      setSending(false);
    }
  };

  // ── Meal card actions ────────────────────────────────────────────────────
  const confirmCoachMemoryFacts = async (facts: CandidateFact[]) => {
    if (!user) throw new Error("no user");
    const ids = await saveCoachMemoryFacts(user.id, facts);
    lastConfirmedFactIdsRef.current = ids;
  };

  const logSuggestedMeal = async (m: SuggestedMeal) => {
    if (!user) throw new Error("no user");
    await logMeal(user.id, {
      food_name: m.name,
      calories: m.calories,
      protein_g: m.protein_g,
      carbs_g: m.carbs_g,
      fat_g: m.fat_g,
      meal_type: (m.meal_type as MealType) ?? inferMealType(),
    });
    invalidateFoodTodayCache(user.id);
    invalidateContextSlice("nutrition");
    grantXP(user.id, XP.FOOD_LOG).catch(() => {});
  };

  const planSuggestedMeal = async (m: SuggestedMeal) => {
    if (!user) throw new Error("no user");
    const t = new Date();
    t.setDate(t.getDate() + 1);
    const planDate = `${t.getFullYear()}-${String(t.getMonth() + 1).padStart(2, "0")}-${String(t.getDate()).padStart(2, "0")}`;
    await replacePlannedMeal(user.id, {
      plan_date: planDate,
      meal_type: (m.meal_type as MealType) ?? inferMealType(),
      food_name: m.name,
      quantity_g: null,
      calories: m.calories,
      protein_g: m.protein_g,
      carbs_g: m.carbs_g,
      fat_g: m.fat_g,
    });
  };

  // Adds every suggested exercise into today's (or the next upcoming) training
  // day on the user's active program. Returns {needsPlan:true} when there's no
  // active program at all, so the card can offer to build one instead.
  const addExercisesToPlan = async (exercises: SuggestedExercise[]): Promise<AddExercisesResult> => {
    if (!user) throw new Error("no user");
    const active = await getActiveProgramWeek(user.id);
    if (!active || active.days.length === 0) return { needsPlan: true };

    const targetDay =
      active.days.find((d) => d.isRecommended && !d.isRest) ??
      active.days.find((d) => !d.isRest) ??
      active.days[0];

    const plan = await getMyPlan(user.id);
    const program = plan?.program as any;
    const workout = (program?.workouts as any[] | undefined)?.find(
      (w) => w.id === targetDay.workoutId
    );
    let orderIndex = workout?.exercises?.length ?? 0;

    for (const ex of exercises) {
      const repsArr = ex.reps.split("/").map((r) => parseInt(r, 10)).filter((n) => !isNaN(n));
      const sets: CustomSet[] = Array.from({ length: Math.max(1, ex.sets) }, (_, i) => ({
        set_number: i + 1,
        reps: repsArr[i] ?? repsArr[repsArr.length - 1] ?? 10,
        weight_kg: null,
      }));
      await addExerciseToWorkout(targetDay.workoutId, ex.name, sets, ex.notes ?? "", orderIndex++);
    }

    invalidateContextSlice("training");
    return { dayTitle: targetDay.title };
  };

  // Append local (non-/chat) messages with a timestamp. `persist` saves them to
  // ai_conversations so meaningful turns survive a reload; transient errors don't.
  const appendLocal = (
    msgs: { role: "user" | "assistant"; content: string }[],
    persist = false
  ) => {
    const stamped: Message[] = msgs.map((m) => ({ ...m, id: newLocalId(), created_at: nowIso() }));
    setMessages((prev) => [...prev, ...stamped]);
    if (persist && user && activeCoach) {
      stamped.forEach((m) =>
        saveMessage(user.id, activeCoach.id, m.role, m.content).catch(() => {})
      );
    }
  };

  const startProgramWizard = async () => {
    appendLocal([{ role: "user", content: "Build me a program" }], false);
    // The map is anatomical, so showing a man's body to a woman is a poor look.
    // One cheap read, before the card mounts, so it never flashes the wrong one.
    if (user) {
      try {
        const { data } = await supabase
          .from("user_preferences")
          .select("gender")
          .eq("user_id", user.id)
          .maybeSingle();
        setBriefGender(String(data?.gender ?? "").toLowerCase() === "female" ? "female" : "male");
      } catch {
        // Default stands. A wrong silhouette is cosmetic; blocking the brief is not.
      }
    }
    setPlanBrief({ step: 0, focus: [], avoid: [], answers: [], generating: false });
  };

  // Multi-select: tapping a muscle toggles it, on whichever list this step owns.
  const toggleBriefMuscle = (slug: string) => {
    setPlanBrief((b) => {
      if (!b) return b;
      const key = b.step === 0 ? "focus" : "avoid";
      const cur = b[key];
      return { ...b, [key]: cur.includes(slug) ? cur.filter((m) => m !== slug) : [...cur, slug] };
    });
  };

  // Skip clears whatever was picked on this step; Continue keeps it.
  const advanceBrief = (clear: boolean) => {
    setPlanBrief((b) => {
      if (!b) return b;
      const key = b.step === 0 ? "focus" : "avoid";
      return { ...b, step: b.step + 1, ...(clear ? { [key]: [] } : {}) };
    });
  };

  const startAdjustWizard = () => {
    appendLocal([{ role: "user", content: "Adjust my plan" }], false);
    setAdjustWizard({ step: 0, answers: [], generating: false });
  };

  const handleAdjustWizardSelect = async (option: string) => {
    if (!adjustWizard) return;
    // Single-question wizard, the choice fires the targeted prompt immediately.
    setAdjustWizard(null);
    await sendMessage(ADJUST_REASON_PROMPTS[option] ?? `Adjust my plan: ${option}. Suggest 2-3 specific changes. Be direct, no questions.`);
  };

  const startCheatWizard = () => {
    appendLocal([{ role: "user", content: "Cheat meal today" }], false);
    setCheatWizard({ step: 0, answers: [], generating: false });
  };

  const handleCheatWizardSelect = async (option: string) => {
    if (!cheatWizard) return;
    const answers = [...cheatWizard.answers, option];
    if (answers.length < CHEAT_WIZARD_QUESTIONS.length) {
      setCheatWizard({ ...cheatWizard, step: answers.length, answers, generating: false });
      return;
    }
    const [slot, kind, intensity] = answers;
    setCheatWizard(null);
    // "Planning a cheat meal" is the phrase ai/rag_pipeline._detect_cheat_plan
    // keys on to force the question intent — without it an 8B classifier reads
    // "I'm eating a burger tonight" as a food log and answers with a log card.
    await sendMessage(
      `Planning a cheat meal, do not log any food. Cheating at: ${slot}. What it'll be: ${kind}. ` +
        `How hard I'm going: ${intensity}. Work from what I've already eaten today and my targets, ` +
        `and tell me how to spend the rest of the day around it.`
    );
  };

  // null closes the field; a string is the live value being typed.
  const handleBriefFreeText = (value: string | null) => {
    if (!planBrief) return;
    setPlanBrief({ ...planBrief, freeText: value });
  };

  const handleBriefSelect = async (option: string) => {
    if (!planBrief) return;
    // Opens the field rather than answering. Guarded on freeText being closed so
    // that typing the literal words "Something else" still submits as an answer.
    if (option === BRIEF_FREE_TEXT_OPT && planBrief.freeText == null) {
      setPlanBrief({ ...planBrief, freeText: "" });
      return;
    }
    const answers = [...planBrief.answers, option];
    if (answers.length < PLAN_BRIEF_MCQ.length) {
      setPlanBrief({
        ...planBrief,
        step: BRIEF_MUSCLE_STEPS + answers.length,
        answers,
        generating: false,
        freeText: null,  // never carry an open field into the next question
      });
    } else {
      setPlanBrief({ ...planBrief, answers, generating: true });
      const label = (m: string) => MUSCLE_LABELS[m] ?? m;
      const [pain, deadline] = answers;
      // Goal, training days and session length are NOT restated here — they are
      // in user_preferences and ride the context. Repeating them invites the
      // model to trust the echo over the profile, and asking for them in the
      // first place is what made the old wizard feel like a form.
      const parts = ["Build me a custom workout program."];
      if (planBrief.focus.length) parts.push(`Prioritise these muscles: ${planBrief.focus.map(label).join(", ")}.`);
      if (planBrief.avoid.length) parts.push(`Avoid or go easy on: ${planBrief.avoid.map(label).join(", ")}.`);
      if (pain && pain !== "Nothing") parts.push(`Currently bothering me: ${pain} — work around it.`);
      if (deadline && deadline !== "Not really") parts.push(`I am working towards something roughly ${deadline} away.`);
      // No trailing "use my profile, do not ask me" instruction here. That is a
      // rule about how to build programs, so it lives in the backend's PROGRAM
      // RULES where it belongs — and keeping it out means this message reads
      // like something a person typed, which matters because the BACKEND is
      // what persists it. Reload the thread and you see these words, not a
      // machine directive.
      const summary: PlanBriefSummary = {
        focus: planBrief.focus.map(label),
        avoid: planBrief.avoid.map(label),
        pain,
        deadline,
        display: BRIEF_DISPLAY,
      };
      setPlanBrief(null);
      await sendMessage(parts.join(" "), { display: BRIEF_DISPLAY, planBrief: summary });
    }
  };

  const startSwapWizard = async () => {
    appendLocal([{ role: "user", content: "Swap an exercise" }], false);
    setSwapTargetExercise(null);
    setSwapPicker({ exercises: [], fetching: true });
    try {
      const [active, plan] = user
        ? await Promise.all([getActiveProgramWeek(user.id), getMyPlan(user.id)])
        : [null, null];
      const todayDay =
        active?.days.find((d) => d.isRecommended && !d.isRest) ??
        active?.days.find((d) => !d.isRest);
      const workout = (plan as any)?.program?.workouts?.find(
        (w: any) => w.id === todayDay?.workoutId
      );
      const exercises: string[] =
        (workout?.exercises as any[] | undefined)?.map((e: any) => e.name).filter(Boolean) ?? [];
      if (exercises.length === 0) {
        // No active plan = nothing to swap. Falling through to the abstract
        // muscle-group wizard made the coach INVENT an exercise the user doesn't
        // have (and the card then offered an unrelated replacement), so stop
        // here and point them at building a plan instead.
        setSwapPicker(null);
        setSwapWizard(null);
        appendLocal(
          [{
            role: "assistant",
            content:
              "You don't have an active plan yet, so there's no exercise to swap. Tap **Build me a program** and once you're training I can swap anything in it.",
          }],
          true
        );
      } else {
        setSwapPicker({ exercises, fetching: false });
      }
    } catch {
      setSwapPicker(null);
      setSwapWizard(null);
      appendLocal(
        [{ role: "assistant", content: "Couldn't load your plan just now — try again in a moment." }],
        true
      );
    }
  };

  const handleSwapPickerSelect = (name: string | null) => {
    setSwapTargetExercise(name);
    setSwapPicker(null);
    setSwapWizard({ step: 0, answers: [], generating: false });
  };

  // Picking the exact exercise already tells us the muscle, so that step is
  // dropped and we only ask why.
  const swapQuestions = swapTargetExercise ? [SWAP_REASON_QUESTION] : SWAP_WIZARD_QUESTIONS;

  const handleSwapWizardSelect = async (option: string) => {
    if (!swapWizard) return;
    const answers = [...swapWizard.answers, option];
    if (answers.length < swapQuestions.length) {
      setSwapWizard({ ...swapWizard, step: answers.length, answers, generating: false });
      return;
    }
    // A skipped step leaves a null — treat that as the neutral "Need variety".
    const reason = answers[answers.length - 1] ?? "Need variety";

    // Targeted swap → offer real alternatives straight from the exercise
    // library (ranked by similarity, re-weighted by the reason). No LLM call:
    // instant, costs nothing, and it can only ever offer exercises that exist.
    if (swapTargetExercise) {
      const original = swapTargetExercise;
      setSwapWizard(null);
      setSwapAlts({ original, options: [], fetching: true });
      try {
        const ctx = user?.id ? await getEligibilityCtx(user.id) : undefined;
        const alts = await getExerciseAlternatives(original, 8, ctx);
        const ranked = rankAlternatives(alts, reason).slice(0, 3);
        if (ranked.length > 0) {
          setSwapAlts({ original, options: ranked, fetching: false });
          return;
        }
      } catch {
        /* fall through to the coach */
      }
      // Nothing in the library matched — let the coach suggest something.
      setSwapAlts(null);
      await sendMessage(
        `I want to swap ${original} in my current workout plan. Reason: ${reason}. Suggest a specific alternative exercise with sets and reps that fits my program.`
      );
      return;
    }

    // No specific target — keep the original coach-decides flow.
    setSwapWizard({ ...swapWizard, answers, generating: true });
    const [muscleGroup] = answers;
    const prompt = `I want to swap a ${muscleGroup} exercise in my current workout plan. Pick the best exercise to replace in my current plan. Reason: ${reason}. Suggest a specific alternative exercise with sets and reps that fits my program.`;
    setSwapWizard(null);
    await sendMessage(prompt);
  };

  // User picked a replacement → hand it to the existing confirm card, which
  // applies it for today only or across the whole plan.
  const chooseSwapAlternative = (replacement: string) => {
    const original = swapAlts?.original;
    setSwapAlts(null);
    if (!original) return;
    haptic.select();
    setPendingAction({
      type: "exercise_swap",
      data: { original_exercise: original, new_exercise: replacement },
    });
  };

  // Log the (possibly edited) macros straight into food_logs. The values shown
  // on the card are exactly what gets saved, no fragile chat round-trip.
  const confirmFoodLog = async (items: FoodConfirmation[]) => {
    if (!pendingAction || pendingAction.type !== "food_log" || !user) return;
    if (foodCardTimerRef.current) clearTimeout(foodCardTimerRef.current);
    setFoodCardReady(false);
    setPendingAction(null);
    try {
      await Promise.all(
        items.map((meal) =>
          logMeal(user.id, {
            food_name: meal.food_name,
            quantity_g: meal.quantity ?? null,
            calories: meal.calories ?? null,
            protein_g: meal.protein_g ?? null,
            carbs_g: meal.carbs_g ?? null,
            fat_g: meal.fats_g ?? null,
            meal_type: (meal.meal_type as MealType) ?? inferMealType(),
          })
        )
      );
      haptic.success();
      if (activeCoach) clearPendingActions(user.id, activeCoach.id).catch(() => {});
      grantXP(user.id, XP.FOOD_LOG).catch(() => {});
      invalidateContextSlice("nutrition");
      post("/invalidate-context", { user_id: user.id, slice: "nutrition" }).catch(() => {});
      fetchLiveUserContext(user.id).then(setLiveContext).catch(() => {});

      const totalCal = items.reduce((s, m) => s + (m.calories ?? 0), 0);
      const totalPro = items.reduce((s, m) => s + (m.protein_g ?? 0), 0);
      const names = items.map((m) => m.food_name).join(" + ");
      const summary = [
        totalCal > 0 ? `${Math.round(totalCal)} kcal` : null,
        totalPro > 0 ? `${Math.round(totalPro)}g protein` : null,
      ].filter(Boolean).join(" · ");
      appendLocal(
        [{
          role: "assistant",
          content: `Logged ${names}${summary ? `, ${summary}` : ""}. Nice work!`,
        }],
        true
      );
    } catch (e) {
      if (__DEV__) console.log("[coach] logMeal error:", e);
      appendLocal([{ role: "assistant", content: "Couldn't log that meal right now. Please try again." }]);
    }
  };

  const cancelFoodLog = () => {
    if (!pendingAction || pendingAction.type !== "food_log") return;
    if (foodCardTimerRef.current) clearTimeout(foodCardTimerRef.current);
    setFoodCardReady(false);
    setPendingAction(null);
    if (user && activeCoach) clearPendingActions(user.id, activeCoach.id).catch(() => {});
    appendLocal([{ role: "assistant", content: "Got it, I won't log that." }], true);
  };

  const confirmWeightLog = async (kg: number) => {
    if (!pendingAction || pendingAction.type !== "weight_log" || !user) return;
    setPendingAction(null);
    try {
      await logBodyWeight(user.id, kg);
      haptic.success();
      grantXP(user.id, XP.WEIGHT_LOG).catch(() => {});
      invalidateContextSlice("progress");
      post("/invalidate-context", { user_id: user.id, slice: "progress" }).catch(() => {});
      fetchLiveUserContext(user.id).then(setLiveContext).catch(() => {});
      appendLocal([{ role: "assistant", content: `Weight logged: ${kg}kg. Keep it up!` }], true);
    } catch {
      appendLocal([{ role: "assistant", content: "Couldn't log your weight right now. Please try again." }]);
    }
  };

  const confirmWaterLog = async (glasses: number) => {
    if (!pendingAction || pendingAction.type !== "water_log" || !user) return;
    setPendingAction(null);
    try {
      const today = new Date();
      const dateStr = `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, "0")}-${String(today.getDate()).padStart(2, "0")}`;
      // Additive: "drank 3 glasses" adds to whatever is already logged today.
      const current = await getWaterGlasses(user.id, dateStr);
      const total = Math.min(20, current + glasses);
      await setWaterGlasses(user.id, dateStr, total);
      haptic.success();
      invalidateContextSlice("nutrition");
      appendLocal(
        [{ role: "assistant", content: `Water logged, you're at ${total}/8 glasses today. ${total >= 8 ? "Hydration goal hit!" : "Keep sipping."}` }],
        true
      );
    } catch {
      appendLocal([{ role: "assistant", content: "Couldn't log your water right now. Please try again." }]);
    }
  };

  const cancelWaterLog = () => {
    if (!pendingAction || pendingAction.type !== "water_log") return;
    setPendingAction(null);
  };

  const confirmCorrection = async (data: CorrectionConfirmation) => {
    if (!pendingAction || pendingAction.type !== "correction" || !user) return;
    setPendingAction(null);
    try {
      // Delete the resolved row(s), then re-log the corrected entry if this was
      // a quantity/macro fix rather than a plain removal.
      await Promise.all(data.items.map((it) => deleteFoodLog(it.id)));
      if (data.corrected) {
        await logMeal(user.id, {
          food_name: data.corrected.food_name,
          quantity_g: data.corrected.quantity_g ?? null,
          calories: data.corrected.calories ?? null,
          protein_g: data.corrected.protein_g ?? null,
          carbs_g: data.corrected.carbs_g ?? null,
          fat_g: data.corrected.fat_g ?? null,
          meal_type: (data.corrected.meal_type as MealType) ?? inferMealType(),
        });
      }
      haptic.success();
      if (activeCoach) clearPendingActions(user.id, activeCoach.id).catch(() => {});
      invalidateFoodTodayCache(user.id);
      invalidateContextSlice("nutrition");
      post("/invalidate-context", { user_id: user.id, slice: "nutrition" }).catch(() => {});
      fetchLiveUserContext(user.id).then(setLiveContext).catch(() => {});
      const names = data.items.map((i) => i.food_name).join(", ");
      const msg = data.corrected
        ? `Updated ${data.corrected.food_name} in today's log.`
        : `Removed ${names || "that entry"} from today's log.`;
      appendLocal([{ role: "assistant", content: msg }], true);
    } catch {
      appendLocal([{ role: "assistant", content: "Couldn't update your log right now. Please try again." }]);
    }
  };

  const cancelCorrection = () => {
    if (!pendingAction || pendingAction.type !== "correction") return;
    setPendingAction(null);
    if (user && activeCoach) clearPendingActions(user.id, activeCoach.id).catch(() => {});
  };

  const logWeightDirect = () => {
    appendLocal([
      { role: "user", content: "Log my weight" },
      { role: "assistant", content: "What's your weight today?" },
    ], false);
    setPendingAction({ type: "weight_log", data: { weight_kg: 0 } });
  };

  const cancelWeightLog = () => {
    if (!pendingAction || pendingAction.type !== "weight_log") return;
    setPendingAction(null);
  };

  const confirmWorkoutLog = async (exercises: WorkoutLogExercise[]) => {
    if (!pendingAction || pendingAction.type !== "workout_log" || !user) return;
    setPendingAction(null);
    try {
      const { newPRs } = await logWorkoutSets(user.id, exercises);
      haptic.success();
      if (newPRs.length > 0) grantXP(user.id, XP.PERSONAL_RECORD * newPRs.length).catch(() => {});
      invalidateContextSlice("training");
      post("/invalidate-context", { user_id: user.id, slice: "training" }).catch(() => {});
      fetchLiveUserContext(user.id).then(setLiveContext).catch(() => {});

      const names = exercises.map((e) => e.exercise).join(", ");
      const prNote = newPRs.length > 0 ? ` New PR on ${newPRs.join(", ")}! 🔥` : "";
      appendLocal([{ role: "assistant", content: `Logged ${names}.${prNote}` }], true);
    } catch {
      appendLocal([{ role: "assistant", content: "Couldn't log that workout right now. Please try again." }]);
    }
  };

  const cancelWorkoutLog = () => {
    if (!pendingAction || pendingAction.type !== "workout_log") return;
    setPendingAction(null);
    appendLocal([{ role: "assistant", content: "No problem, nothing logged." }], true);
  };

  // Executed locally, the backend has no exercise_swap executor, and a text
  // round trip ("Just today") depends on the classifier guessing right. The
  // card already carries both names, so we apply the swap directly.
  const confirmExerciseSwap = async (scope: "today" | "always" | "cancel") => {
    if (!pendingAction || pendingAction.type !== "exercise_swap") return;
    const { original_exercise, new_exercise } = pendingAction.data;
    setPendingAction(null);
    if (user && activeCoach) clearPendingActions(user.id, activeCoach.id).catch(() => {});

    if (scope === "cancel") {
      appendLocal(
        [
          { role: "user", content: "Cancel" },
          { role: "assistant", content: "No problem, keeping your original exercise." },
        ],
        true
      );
      return;
    }
    if (!original_exercise || !new_exercise || !user) return;

    if (scope === "today") {
      try {
        await addTodaySwap(original_exercise, new_exercise);
        appendLocal(
          [
            { role: "user", content: "Just today" },
            { role: "assistant", content: `Done, today's session will show **${new_exercise}** instead of ${original_exercise}. Your plan stays unchanged.` },
          ],
          true
        );
      } catch {
        appendLocal([{ role: "assistant", content: "Couldn't set that up right now. Please try again." }]);
      }
      return;
    }

    // scope === "always", rename the exercise across the active plan
    try {
      const count = await swapExerciseInActivePlan(user.id, original_exercise, new_exercise);
      invalidateContextSlice("training");
      post("/invalidate-context", { user_id: user.id, slice: "training" }).catch(() => {});
      appendLocal(
        [
          { role: "user", content: "Always replace it" },
          { role: "assistant", content: `Swapped **${original_exercise}** → **${new_exercise}** in ${count} spot${count !== 1 ? "s" : ""} across your plan. Locked in.` },
        ],
        true
      );
    } catch (e: any) {
      appendLocal([
        { role: "assistant", content: e?.message ?? "Couldn't update your plan right now. Please try again." },
      ]);
    }
  };

  const confirmDietPreference = async (scope: "today" | "permanently" | "cancel") => {
    if (!pendingAction || pendingAction.type !== "diet_preference") return;
    const { food_item } = pendingAction.data;
    setPendingAction(null);

    if (scope === "cancel") {
      appendLocal([{ role: "assistant", content: "No problem, noted." }], true);
    } else if (scope === "today") {
      appendLocal([{ role: "assistant", content: `Got it, skipping ${food_item} just for today.` }], true);
    } else if (scope === "permanently") {
      try {
        if (user) {
          await addToAllergies(user.id, food_item);
          invalidateContextSlice("nutrition");
          post("/invalidate-context", { user_id: user.id, slice: "nutrition" }).catch(() => {});
          setPreferences((prev: any) => {
            const current: string[] = Array.isArray(prev?.allergies_or_dislikes)
              ? prev.allergies_or_dislikes
              : [];
            const normalized = food_item.toLowerCase().trim();
            if (current.some((x: string) => x.toLowerCase().trim() === normalized)) return prev;
            return { ...prev, allergies_or_dislikes: [...current, food_item] };
          });
          appendLocal([{ role: "assistant", content: `Done, I'll never suggest ${food_item} again.` }], true);
        }
      } catch (err) {
        if (__DEV__) console.log("[coach] addToAllergies error:", err);
        appendLocal([{ role: "assistant", content: "Couldn't save that preference right now. Try again later." }]);
      }
    }
  };

  // ─── Loading ───────────────────────────────────────────────────────────────

  const chatItems = useMemo(() => buildChatItems(messages), [messages]);

  if (loading) {
    return (
      <SafeAreaView style={s.container}>
        <View style={{ flexDirection: "row", alignItems: "center", gap: 12, padding: 16 }}>
          <Skeleton width={44} height={44} borderRadius={22} />
          <View style={{ gap: 6 }}>
            <Skeleton width={110} height={15} borderRadius={4} />
            <Skeleton width={90} height={12} borderRadius={4} />
          </View>
        </View>
        <View style={{ padding: 16, gap: 12, flex: 1, justifyContent: "flex-end" }}>
          <Skeleton width="70%" height={44} borderRadius={20} style={{ borderBottomLeftRadius: 6, alignSelf: "flex-start" }} />
          <Skeleton width="55%" height={36} borderRadius={20} style={{ borderBottomLeftRadius: 6, alignSelf: "flex-start" }} />
          <Skeleton width="80%" height={60} borderRadius={20} style={{ borderBottomLeftRadius: 6, alignSelf: "flex-start" }} />
          <Skeleton width="45%" height={36} borderRadius={20} style={{ borderBottomLeftRadius: 6, alignSelf: "flex-start" }} />
        </View>
      </SafeAreaView>
    );
  }

  if (coaches.length === 0) {
    return (
      <SafeAreaView style={s.container}>
        <ScrollView keyboardDismissMode={KEYBOARD_DISMISS} contentContainerStyle={s.pickerScroll} showsVerticalScrollIndicator={false}>
          <FadeSlideIn delay={0}>
            <View style={s.pickerIconWrap}>
              <Ionicons name="sparkles" size={26} color={COLORS.primary} />
            </View>
            <Text style={s.pickerTitle}>Choose your AI coach</Text>
            <Text style={s.pickerSub}>
              Pick a coach to unlock personalized AI training & nutrition, built on their style.
            </Text>
          </FadeSlideIn>

          {suggestedCreators.length === 0 ? (
            <ActivityIndicator color={COLORS.primary} style={{ marginTop: 40 }} />
          ) : (
            <FadeSlideIn delay={80}>
              <Text style={s.pickerSectionLabel}>SUGGESTED FOR YOU</Text>
              <View style={{ marginTop: 10, gap: 12 }}>
                {suggestedCreators.map((c) => (
                  <TouchableOpacity
                    key={c.id}
                    style={s.creatorCard}
                    activeOpacity={0.85}
                    onPress={() => { haptic.select(); setConfirmCreator(c); }}
                  >
                    {c.profile_image ? (
                      <Image source={{ uri: c.profile_image }} style={s.creatorAvatar} contentFit="cover" />
                    ) : (
                      <View style={[s.creatorAvatar, s.creatorAvatarFallback]}>
                        <Text style={s.creatorAvatarInitial}>{c.name?.[0]?.toUpperCase() ?? "C"}</Text>
                      </View>
                    )}
                    <View style={{ flex: 1, minWidth: 0 }}>
                      <Text style={s.creatorName} numberOfLines={1}>{c.name}</Text>
                      {!!c.specialty && (
                        <Text style={s.creatorSpecialty} numberOfLines={1}>{c.specialty}</Text>
                      )}
                      {(c.total_subscribers ?? 0) > 0 && (
                        <Text style={s.creatorSubs}>{c.total_subscribers} subscribers</Text>
                      )}
                    </View>
                    <View style={s.creatorSelectBtn}>
                      <Text style={s.creatorSelectTxt}>Select</Text>
                    </View>
                  </TouchableOpacity>
                ))}
              </View>

              <TouchableOpacity
                style={s.browseAllBtn}
                onPress={() => { haptic.light(); router.push("/(tabs)/search" as never); }}
                activeOpacity={0.85}
              >
                <Ionicons name="compass-outline" size={18} color={COLORS.primary} />
                <Text style={s.browseAllTxt}>Explore all coaches</Text>
                <Ionicons name="chevron-forward" size={16} color={COLORS.primary} style={{ marginLeft: "auto" }} />
              </TouchableOpacity>
            </FadeSlideIn>
          )}
        </ScrollView>

        <ConfirmDialog
          visible={!!confirmCreator}
          icon="sparkles"
          title={`Coach with ${confirmCreator?.name ?? "this coach"}?`}
          message={`Your AI coach will be built on ${confirmCreator?.name ?? "their"}'s training & nutrition style. You can switch after 30 days.`}
          confirmText={subscribing ? "Selecting…" : "Confirm"}
          onConfirm={handleConfirmSelectCreator}
          onCancel={() => { if (!subscribing) setConfirmCreator(null); }}
        />
      </SafeAreaView>
    );
  }

  // ─── Main chat ─────────────────────────────────────────────────────────────

  return (
    <KeyboardAvoidingView
      style={{ flex: 1 }}
      behavior={Platform.OS === "ios" ? "padding" : "height"}
      keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
    >
      {/* edges: no "bottom".
          This screen sits inside the tab navigator, so its frame already stops
          above the tab bar and the tab bar is what covers the home indicator.
          A bottom safe-area inset here is a SECOND helping of the same gap: with
          the keyboard up it stacked insets.bottom (34pt on a notched phone) on
          top of the inputBar's own padding and floated the composer ~58pt above
          the keyboard instead of resting on it. */}
      <SafeAreaView style={s.container} edges={["top", "left", "right"]}>

        {/* Coach header. Overlaid rather than stacked so the conversation
            scrolls beneath it; flat black since the redesign dropped glass. */}
        <FadeSlideIn delay={0} style={s.coachHeaderWrap}>
        <TouchableOpacity
          onLayout={(e) => setHeaderH(e.nativeEvent.layout.height)}
          style={s.coachHeader}
          onPress={() => {
            if (coaches.length > 1) {
              haptic.select();
              setCoachPickerOpen(true);
            }
          }}
          activeOpacity={coaches.length > 1 ? 0.7 : 1}
        >
          {/* Avatar with lime ring */}
          <View style={s.coachAvatarRing}>
            {activeCoach?.profile_image ? (
              <Image source={{ uri: activeCoach.profile_image }} style={s.coachAvatar} />
            ) : (
              <View style={s.coachAvatarFallback}>
                <Text style={s.coachAvatarInitial}>{activeCoach?.name?.[0] ?? "C"}</Text>
              </View>
            )}
          </View>

          <View style={{ flex: 1 }}>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 6 }}>
              <Text style={s.coachName}>{activeCoach?.name}</Text>
              {coaches.length > 1 && <Text style={s.coachSwitchHint}>▾</Text>}
            </View>
            <Text style={s.coachSpecialty}>Knows your plan, food and history</Text>
          </View>

          {messagesRemaining != null && !upgradeRequired && (
            <Text style={s.msgCounterTxt}>{messagesRemaining} / 20</Text>
          )}

          {upgradeRequired && (
            <Text style={[s.msgCounterTxt, { color: "#FF4444" }]}>Limit reached</Text>
          )}
        </TouchableOpacity>
        </FadeSlideIn>

        {/* Messages, watermark only in the empty state so it never sits under bubbles */}
        <View style={s.messagesRegion}>
          {messages.length === 0 && <ChatBackdrop coach={activeCoach} />}
          <ScrollView
            ref={scrollRef}
            style={{ flex: 1 }}
            showsVerticalScrollIndicator={false}
            keyboardShouldPersistTaps="handled"
            keyboardDismissMode={Platform.OS === "ios" ? "interactive" : "on-drag"}
            contentContainerStyle={[s.messagesContent, { paddingTop: headerH + 16 }]}
            onScroll={onScroll}
            scrollEventThrottle={32}
            onContentSizeChange={() => {
              // Once content is measured, honor a pending open-at-latest intent.
              if (wantBottomRef.current) {
                wantBottomRef.current = false;
                scrollRef.current?.scrollToEnd({ animated: false });
              }
            }}
            maintainVisibleContentPosition={{ minIndexForVisible: 1 }}
          >
          {/* index 0, load-older header (keeps prepended batches from jumping) */}
          <View style={s.loadOlderHeader}>
            {loadingMore ? (
              <ActivityIndicator color={COLORS.primary} size="small" />
            ) : !hasMore && messages.length > HISTORY_END_MIN ? (
              <Text style={s.historyEndTxt}>You've reached the beginning of this conversation</Text>
            ) : hasMore && messages.length > 0 ? (
              <TouchableOpacity onPress={() => { haptic.select(); loadOlder(); }} style={s.loadOlderBtn}>
                <Text style={s.loadOlderBtnTxt}>Load older messages</Text>
              </TouchableOpacity>
            ) : null}
          </View>

          {messages.length === 0 && (
            <Animated.View entering={FadeInUp.duration(440).delay(140).easing(Easing.out(Easing.cubic))}>
              <View style={s.daySep}>
                <View style={s.daySepLine} />
                <Text style={s.daySepTxt}>Today</Text>
                <View style={s.daySepLine} />
              </View>
              <View style={s.bubbleAssistant}>
                <Text style={s.bubbleTxt}>
                  Hey {profile?.full_name?.split(" ")[0] ?? "there"}! I'm{" "}
                  {activeCoach?.name}. Ask me anything about your training,
                  nutrition, or recovery. I'm here to help you crush your goals.
                </Text>
              </View>
            </Animated.View>
          )}

          {messages.length > 0 && chatItems.map((item, idx) => {
            if (item.kind === "day") {
              return (
                <View key={item.key} style={s.daySep}>
                  <View style={s.daySepLine} />
                  <Text style={s.daySepTxt}>{item.label}</Text>
                  <View style={s.daySepLine} />
                </View>
              );
            }
            const isUser = item.msg.role === "user";
            // Staggered rise-in on mount; capped so late messages don't lag.
            // Suppressed during scroll-up prepends (handled by the ref).
            const entering = suppressEnterRef.current
              ? undefined
              : FadeInUp.duration(360).delay(Math.min(idx, 6) * 45).easing(Easing.out(Easing.cubic));
            return (
              <Animated.View key={item.key} entering={entering}>
                <View style={isUser ? s.bubbleUser : s.bubbleAssistant}>
                  {!isUser && item.msg.animate && !revealedIdsRef.current.has(item.msg.id) ? (
                    <TypewriterMdText
                      style={s.bubbleTxt}
                      text={item.msg.content}
                      onTick={() => scrollRef.current?.scrollToEnd({ animated: false })}
                      onDone={() => {
                        revealedIdsRef.current.add(item.msg.id);
                        scrollRef.current?.scrollToEnd({ animated: true });
                      }}
                    />
                  ) : (
                    <MdText
                      style={[s.bubbleTxt, isUser && s.bubbleTxtUser]}
                      text={item.msg.content}
                    />
                  )}
                  {item.msg.created_at ? (
                    <Text style={[s.bubbleTime, isUser && s.bubbleTimeUser]}>
                      {timeLabel(item.msg.created_at)}
                    </Text>
                  ) : null}
                </View>
                {item.msg.role === "assistant" && item.msg.programProposal &&
                  pendingAction?.type !== "program_proposal" && (
                  <ProgramProposalCard data={item.msg.programProposal} frozen />
                )}
                {item.msg.role === "assistant" && (item.msg.exercises?.length ?? 0) > 0 && (
                  <View style={s.exerciseCardsWrap}>
                    {item.msg.exercises!.map((ex) => (
                      <ChatExerciseCard
                        key={ex.id}
                        exercise={ex}
                        onPress={() => setSelectedExercise(ex)}
                      />
                    ))}
                  </View>
                )}
                {item.msg.role === "assistant" && (item.msg.meals?.length ?? 0) > 0 && (
                  <SuggestedMealCards
                    meals={item.msg.meals!}
                    onLog={logSuggestedMeal}
                    onPlan={planSuggestedMeal}
                  />
                )}
                {item.msg.role === "assistant" && (item.msg.workoutExercises?.length ?? 0) > 0 && (
                  <WorkoutSuggestionCard
                    exercises={item.msg.workoutExercises!}
                    onAdd={() => addExercisesToPlan(item.msg.workoutExercises!)}
                    onBuildPlan={startProgramWizard}
                  />
                )}
                {item.msg.planBrief && <BriefSummaryCard data={item.msg.planBrief} />}
                {item.msg.role === "assistant" && (item.msg.candidateFacts?.length ?? 0) > 0 && (
                  <CoachMemoryCard
                    facts={item.msg.candidateFacts!}
                    onConfirm={confirmCoachMemoryFacts}
                  />
                )}
                {item.msg.role === "assistant" && item.msg.progressCard && user && (
                  <TodayProgressCard userId={user.id} />
                )}
                {item.msg.role === "assistant" && item.msg.weeklyCard && user && (
                  <WeeklyStatsCard userId={user.id} />
                )}
              </Animated.View>
            );
          })}

          {/* Pending confirmation cards */}
          {pendingAction?.type === "food_log" && (
            <FadeSlideIn key={`food-${foodCardReady}`}>
              {foodCardReady ? (
                <FoodLogCard
                  data={pendingAction.data}
                  onConfirm={confirmFoodLog}
                  onCancel={cancelFoodLog}
                />
              ) : (
                <View style={s.bubbleAssistant}>
                  <Text style={s.bubbleTxt}>
                    Fetching macros... tap any field to edit before logging.
                  </Text>
                </View>
              )}
            </FadeSlideIn>
          )}

          {pendingAction?.type === "weight_log" && (
            <FadeSlideIn>
              <WeightLogCard
                weight_kg={pendingAction.data.weight_kg}
                onConfirm={confirmWeightLog}
                onCancel={cancelWeightLog}
              />
            </FadeSlideIn>
          )}

          {pendingAction?.type === "water_log" && (
            <FadeSlideIn>
              <WaterLogCard
                glasses={pendingAction.data.glasses}
                onConfirm={confirmWaterLog}
                onCancel={cancelWaterLog}
              />
            </FadeSlideIn>
          )}

          {pendingAction?.type === "workout_log" && (
            <FadeSlideIn>
              <WorkoutLogCard
                data={pendingAction.data}
                onConfirm={confirmWorkoutLog}
                onCancel={cancelWorkoutLog}
              />
            </FadeSlideIn>
          )}

          {pendingAction?.type === "correction" && (
            <FadeSlideIn>
              <CorrectionCard
                data={pendingAction.data}
                onConfirm={confirmCorrection}
                onCancel={cancelCorrection}
              />
            </FadeSlideIn>
          )}

          {pendingAction?.type === "exercise_swap" && (
            <FadeSlideIn>
              <ExerciseSwapCard
                data={pendingAction.data}
                onToday={() => confirmExerciseSwap("today")}
                onAlways={() => confirmExerciseSwap("always")}
                onNo={() => confirmExerciseSwap("cancel")}
              />
            </FadeSlideIn>
          )}

          {pendingAction?.type === "program_proposal" && (
            <FadeSlideIn>
              <ProgramProposalCard
                data={pendingAction.data}
                onActivate={() => { setPendingAction(null); sendMessage("yes"); }}
                onCancel={() => { setPendingAction(null); sendMessage("no"); }}
                getAlternatives={async (name) => {
                  const ctx = user?.id ? await getEligibilityCtx(user.id) : undefined;
                  const alts = await getExerciseAlternatives(name, 4, ctx);
                  return alts.map((a) => ({
                    name: a.name,
                    muscle: (a.primary_muscles?.[0] ?? a.muscle_group ?? "").replace(/_/g, " "),
                  }));
                }}
                onSwap={async (updated) => {
                  setPendingAction({ type: "program_proposal", data: updated });
                  if (!user || !activeCoach) return;
                  const { isModify, ...args } = updated;
                  try {
                    await updatePendingProposal(user.id, activeCoach.id, {
                      tool: "propose_workout_program",
                      args,
                      intent: isModify ? "modify_program" : "create_program",
                    });
                  } catch { /* swap still reflected locally; retried on next swap */ }
                }}
                onAddToPlan={async () => {
                  if (!user) return;
                  const d = pendingAction.data;
                  try {
                    await saveProgramFromProposal(
                      user.id,
                      { title: d.title, difficulty: d.difficulty, goal: d.goal, duration_weeks: d.duration_weeks, days: d.days },
                      { activate: false }
                    );
                    if (activeCoach) await clearPendingProposal(user.id, activeCoach.id).catch(() => {});
                    setPendingAction(null);
                    invalidateContextSlice("training");
                    appendLocal([{ role: "assistant", content: `Saved **${d.title}** to your workout plans. Switch to it anytime from the Workout tab.` }], true);
                  } catch {
                    appendLocal([{ role: "assistant", content: "Couldn't save that to your plans just now, please try again." }]);
                  }
                }}
              />
            </FadeSlideIn>
          )}

          {pendingAction?.type === "diet_preference" && (
            <FadeSlideIn>
              <DietPreferenceCard
                data={pendingAction.data}
                onJustToday={() => confirmDietPreference("today")}
                onPermanently={() => confirmDietPreference("permanently")}
                onCancel={() => confirmDietPreference("cancel")}
              />
            </FadeSlideIn>
          )}

          {planBrief && (
            <FadeSlideIn>
              <PlanBriefCard
                brief={planBrief}
                onFreeText={handleBriefFreeText}
                gender={briefGender}
                onToggleMuscle={toggleBriefMuscle}
                onNext={() => advanceBrief(false)}
                onSkip={() => advanceBrief(true)}
                onSelect={handleBriefSelect}
              />
            </FadeSlideIn>
          )}

          {swapPicker && (
            <FadeSlideIn>
              <ExercisePickerCard
                exercises={swapPicker.exercises}
                fetching={swapPicker.fetching}
                onPick={handleSwapPickerSelect}
                onSkip={() => handleSwapPickerSelect(null)}
              />
            </FadeSlideIn>
          )}

          {swapAlts && (
            <FadeSlideIn>
              <SwapAlternativesCard
                original={swapAlts.original}
                options={swapAlts.options}
                fetching={swapAlts.fetching}
                gender={preferences?.gender?.toLowerCase() === "female" ? "female" : "male"}
                onPick={chooseSwapAlternative}
                onCancel={() => setSwapAlts(null)}
              />
            </FadeSlideIn>
          )}

          {swapWizard && (
            <FadeSlideIn>
              <WizardCard
                wizard={swapWizard}
                questions={swapQuestions}
                generatingLabel="Finding the best swap..."
                onSelect={handleSwapWizardSelect}
              />
            </FadeSlideIn>
          )}

          {adjustWizard && (
            <FadeSlideIn>
              <WizardCard
                wizard={adjustWizard}
                questions={ADJUST_WIZARD_QUESTIONS}
                onSelect={handleAdjustWizardSelect}
              />
            </FadeSlideIn>
          )}

          {cheatWizard && (
            <FadeSlideIn>
              <WizardCard
                wizard={cheatWizard}
                questions={CHEAT_WIZARD_QUESTIONS}
                onSelect={handleCheatWizardSelect}
              />
            </FadeSlideIn>
          )}

          {sending && (
            <Animated.View entering={FadeInUp.duration(260)} style={s.bubbleAssistant}>
              <Text style={s.loadingMsgTxt}>{COACH_LOADING_MSGS[loadingMsgIdx]}</Text>
              <TypingDots />
            </Animated.View>
          )}

          {upgradeRequired && (
            <View style={s.upgradeCard}>
              <Text style={s.upgradeTxt}>{upgradeMessage}</Text>
              <TouchableOpacity
                style={s.upgradeBtn}
                activeOpacity={0.85}
                onPress={() => { haptic.medium(); router.push("/profile/subscription" as never); }}
              >
                <Text style={s.upgradeBtnTxt}>Upgrade to Premium →</Text>
              </TouchableOpacity>
            </View>
          )}
          </ScrollView>
        </View>

        {/* Quick prompts, hidden when a confirmation card or wizard is open */}
        {!pendingAction && !planBrief && !swapWizard && !swapAlts && !adjustWizard && !cheatWizard && !upgradeRequired && (
          <GestureDetector gesture={Gesture.Native().blocksExternalGesture(tabSwipeRef)}>
          <ScrollView keyboardDismissMode={KEYBOARD_DISMISS}
            horizontal
            showsHorizontalScrollIndicator={false}
            decelerationRate="fast"
            style={s.quickPromptsScroll}
            contentContainerStyle={{ paddingHorizontal: 16, paddingVertical: 6, gap: 8 }}
          >
            {QUICK_PROMPTS.map((prompt, i) => (
              <PressableScale
                key={prompt}
                style={i === 0 ? s.quickPromptPrimary : s.quickPrompt}
                haptics="none"
                onPress={() => {
                  if (prompt === "Build me a program") return startProgramWizard();
                  if (prompt === "Swap an exercise") return startSwapWizard();
                  if (prompt === "Adjust my plan") return startAdjustWizard();
                  if (prompt === "Cheat meal today") return startCheatWizard();
                  if (prompt === "Log my weight") return logWeightDirect();
                  sendMessage(PROMPT_OVERRIDES[prompt] ?? prompt);
                }}
                disabled={sending}
              >
                <Text style={i === 0 ? s.quickPromptTxtPrimary : s.quickPromptTxt}>{prompt}</Text>
              </PressableScale>
            ))}
          </ScrollView>
          </GestureDetector>
        )}

        {/* Input bar */}
        {/* Clears the tab bar only while the keyboard is down. With the keyboard
            up the bar sits behind it, and allowing for both floats the composer
            off the keyboard by the tab bar's whole height. */}
        <View style={[s.inputBar, { paddingBottom: keyboardUp ? 12 : 12 + tabBarHeight }]}>
          <Animated.View style={micPulseStyle}>
            <PressableScale
              style={[s.micBtn, isListening && s.micBtnActive]}
              haptics="none"
              hitSlop={{ top: 8, bottom: 8, left: 8, right: 4 }}
              onPress={toggleVoiceInput}
              disabled={upgradeRequired || sending}
              accessibilityLabel={isListening ? "Stop voice input" : "Start voice input"}
              accessibilityRole="button"
            >
              <Ionicons
                name={isListening ? "stop" : "mic"}
                size={20}
                color={isListening ? "#000" : COLORS.textSecondary}
              />
            </PressableScale>
          </Animated.View>
          <TextInput
            style={[s.input, inputFocused && s.inputFocused]}
            onFocus={() => setInputFocused(true)}
            onBlur={() => setInputFocused(false)}
            value={input}
            onChangeText={(text) => {
              // Hardware/IME "return" key inserts a literal "\n" into a multiline
              // TextInput on both platforms before (or instead of) firing
              // onSubmitEditing, treat a trailing newline as "send".
              if (text.endsWith("\n") && text.trim().length > 0) {
                // Sync state BEFORE sending. This used to return without
                // touching state, so `input` held the pre-Enter text while the
                // native box held text + the newline — and any path where
                // sendMessage bails early (most often: a reply is still in
                // flight, so it returns on `sending`) left that newline sitting
                // there with nothing to clear it. On a send that proceeds,
                // sendMessage's own setInput("") still empties the box.
                setInput(text.slice(0, -1));
                submitUserText(text.slice(0, -1));
                return;
              }
              setInput(text);
            }}
            placeholder={
              upgradeRequired
                ? "Daily limit reached — try again tomorrow"
                : isListening
                ? "Listening…"
                : `Ask ${activeCoach?.name ?? "your coach"}…`
            }
            placeholderTextColor={COLORS.textMuted}
            multiline
            maxLength={500}
            returnKeyType="send"
            blurOnSubmit={false}
            onSubmitEditing={() => submitUserText()}
            editable={!upgradeRequired}
          />
          <PressableScale
            style={[s.sendBtn, (!input.trim() || sending || upgradeRequired) && s.sendBtnDisabled]}
            haptics="none"
            hitSlop={{ top: 8, bottom: 8, left: 4, right: 8 }}
            onPress={() => submitUserText()}
            disabled={!input.trim() || sending || upgradeRequired}
            accessibilityLabel="Send message"
            accessibilityRole="button"
          >
            {sending ? (
              <ActivityIndicator color="#000" size="small" />
            ) : (
              <Ionicons
                name="arrow-up"
                size={20}
                color={!input.trim() || upgradeRequired ? COLORS.textFaint : "#000"}
              />
            )}
          </PressableScale>
        </View>

      </SafeAreaView>

      {/* Exercise detail modal */}
      <ExerciseDetailModal
        exercise={selectedExercise}
        gender={preferences?.gender?.toLowerCase() === "female" ? "female" : "male"}
        userId={user?.id}
        onClose={() => setSelectedExercise(null)}
      />

      {/* Coach switcher modal */}
      <DragSheet
        visible={coachPickerOpen}
        onClose={() => setCoachPickerOpen(false)}
        sheetStyle={s.modalSheet}
        backdropColor={s.modalOverlay.backgroundColor}
      >
            <Text style={s.modalTitle}>Switch Coach</Text>
            <Text style={s.modalSub}>
              Each coach has their own style and memory of your conversations.
            </Text>

            {coaches.map((coach) => {
              const isActive = coach.id === activeCoach?.id;
              return (
                <TouchableOpacity
                  key={coach.id}
                  style={[s.coachOption, isActive && s.coachOptionActive]}
                  onPress={() => { haptic.select(); switchCoach(coach); }}
                >
                  {coach.profile_image ? (
                    <Image source={{ uri: coach.profile_image }} style={s.coachOptionImg} />
                  ) : (
                    <View style={[s.coachOptionImg, s.coachOptionImgFallback]}>
                      <Text style={{ color: "#FFF", fontFamily: FONTS.bold }}>{coach.name?.[0]}</Text>
                    </View>
                  )}
                  <View style={{ flex: 1 }}>
                    <Text style={s.coachOptionName}>{coach.name}</Text>
                    <Text style={s.coachOptionSpecialty}>{coach.specialty}</Text>
                  </View>
                  {isActive && <Text style={s.coachOptionCheck}>✓</Text>}
                </TouchableOpacity>
              );
            })}
      </DragSheet>

    </KeyboardAvoidingView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#000" },

  // ── Messages region + backdrop ────────────────────────────────────────────────
  messagesRegion: { flex: 1, position: "relative" },
  messagesContent: { padding: 16, paddingBottom: 24, gap: 12, flexGrow: 1, justifyContent: "flex-end" },

  backdrop: {
    position: "absolute",
    top: 0, left: 0, right: 0, bottom: 0,
    alignItems: "center",
    justifyContent: "center",
    gap: 8,
    paddingBottom: 60,
  },
  backdropRing: {
    width: 104,
    height: 104,
    borderRadius: 52,
    borderWidth: 1,
    borderColor: "rgba(216,255,42,0.14)",
    backgroundColor: "rgba(216,255,42,0.03)",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
    marginBottom: 6,
  },
  backdropAvatar: { width: 104, height: 104, opacity: 0.14 },
  backdropKicker: {
    color: "rgba(255,255,255,0.10)",
    fontFamily: FONTS.bold,
    fontSize: 11,
    letterSpacing: 3,
  },
  backdropName: {
    color: "rgba(255,255,255,0.13)",
    fontFamily: FONTS.black,
    fontSize: 28,
    letterSpacing: -0.5,
  },
  backdropTagline: {
    color: "rgba(255,255,255,0.07)",
    fontFamily: FONTS.medium,
    fontSize: 12.5,
    letterSpacing: 0.5,
  },

  // ── Header ──────────────────────────────────────────────────────────────────
  // The positioned layer. zIndex keeps it over the scroll view, and the
  // hairline is the light catching the lip of the material.
  coachHeaderWrap: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    zIndex: 10,
    backgroundColor: "#000",
    borderBottomWidth: StyleSheet.hairlineWidth,
    borderBottomColor: HAIRLINE,
  },
  coachHeader: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 18,
    paddingTop: 12,
    paddingBottom: 14,
    gap: 12,
  },

  coachAvatarRing: {
    width: 44,
    height: 44,
    borderRadius: 22,
    borderWidth: 2,
    borderColor: "#FFF",
    justifyContent: "center",
    alignItems: "center",
  },
  coachAvatar: { width: 38, height: 38, borderRadius: 19 },
  coachAvatarFallback: {
    width: 38,
    height: 38,
    borderRadius: 19,
    backgroundColor: "#1A1A1C",
    justifyContent: "center",
    alignItems: "center",
  },
  coachAvatarInitial: { color: "#FFF", fontSize: 16, fontFamily: FONTS.extrabold },
  coachName: { color: "#FFF", fontSize: 20, fontFamily: FONTS.extrabold, letterSpacing: -0.3 },
  coachSwitchHint: { color: COLORS.primary, fontSize: 14, fontFamily: FONTS.regular },
  coachSpecialty: { color: COLORS.textMuted, fontSize: 13, fontFamily: FONTS.regular, marginTop: 1 },

  msgCounterTxt: {
    color: COLORS.textMuted,
    fontSize: 12,
    fontFamily: FONTS.medium,
  },

  // ── Bubbles ──────────────────────────────────────────────────────────────────
  // The coach speaks in plain text; only your messages sit in a bubble. Fewer
  // boxes, and the cards the coach attaches stand out as the actions.
  bubbleAssistant: {
    paddingVertical: 4,
    paddingHorizontal: 2,
    maxWidth: "92%",
    alignSelf: "flex-start",
  },
  bubbleUser: {
    backgroundColor: "#1F1F22",
    borderRadius: 20,
    borderBottomRightRadius: 6,
    paddingVertical: 11,
    paddingHorizontal: 14,
    maxWidth: "80%",
    alignSelf: "flex-end",
  },
  bubbleTxt: { color: COLORS.textSecondary, fontSize: 15, lineHeight: 22, fontFamily: FONTS.regular },
  bubbleTxtUser: { color: "#FFF" },
  bubbleTime: { color: COLORS.textMuted, fontSize: 11, fontFamily: FONTS.medium, marginTop: 4, alignSelf: "flex-start" },
  bubbleTimeUser: { color: COLORS.textMuted, alignSelf: "flex-end" },

  // ── Exercise cards (below assistant bubbles) ────────────────────────────────
  exerciseCardsWrap: { alignSelf: "flex-start", gap: 6, marginTop: 2, maxWidth: "85%" },

  // ── Pagination header ────────────────────────────────────────────────────────
  loadOlderHeader: { minHeight: 4, alignItems: "center", justifyContent: "center", paddingVertical: 4 },
  historyEndTxt: { color: COLORS.textFaint, fontFamily: FONTS.medium, fontSize: 11, paddingVertical: 6 },
  loadOlderBtn: { paddingVertical: 6, paddingHorizontal: 14, borderRadius: 12, borderWidth: 1, borderColor: "#222" },
  loadOlderBtnTxt: { color: COLORS.textMuted, fontFamily: FONTS.medium, fontSize: 12 },

  // ── Day separator ────────────────────────────────────────────────────────────
  daySep: { flexDirection: "row", alignItems: "center", gap: 10, marginVertical: 4, paddingHorizontal: 4 },
  daySepLine: { flex: 1, height: 1, backgroundColor: "#1A1A1A" },
  daySepTxt: {
    color: COLORS.textMuted,
    fontFamily: FONTS.medium,
    fontSize: 12,
  },

  // ── Typing indicator ─────────────────────────────────────────────────────────
  typingIndicator: { flexDirection: "row", gap: 4, alignItems: "center", padding: 4 },
  typingDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: COLORS.primary },
  loadingMsgTxt: { color: COLORS.primary, fontFamily: FONTS.medium, fontSize: 13, marginBottom: 6 },

  // ── Upgrade card ─────────────────────────────────────────────────────────────
  upgradeCard: {
    backgroundColor: "#0D0D0D",
    borderRadius: 20,
    padding: 18,
    borderWidth: 0.5,
    borderColor: "#FF444433",
    alignSelf: "stretch",
    marginTop: 4,
  },
  upgradeTxt: { color: "#555555", fontFamily: FONTS.regular, fontSize: 14, marginBottom: 14 },
  upgradeBtn: {
    backgroundColor: COLORS.primary,
    borderRadius: 14,
    height: 52,
    justifyContent: "center",
    alignItems: "center",
  },
  upgradeBtnTxt: { color: "#000", fontFamily: FONTS.bold, fontSize: 15 },

  // ── Quick prompts ────────────────────────────────────────────────────────────
  quickPromptsScroll: { maxHeight: 56, marginBottom: 4 },
  quickPrompt: {
    backgroundColor: "transparent",
    borderRadius: 99,
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },
  quickPromptTxt: { color: "rgba(255,255,255,0.85)", fontSize: 13.5, fontFamily: FONTS.medium },
  quickPromptPrimary: {
    backgroundColor: COLORS.primary,
    borderRadius: 99,
    paddingHorizontal: 18,
    paddingVertical: 10,
  },
  quickPromptTxtPrimary: { color: "#000", fontSize: 13.5, fontFamily: FONTS.bold },

  // ── Input bar ────────────────────────────────────────────────────────────────
  inputBar: {
    flexDirection: "row",
    alignItems: "flex-end",
    paddingHorizontal: 16,
    // Sits directly above the tab bar (or the keyboard), neither of which needs
    // a safe-area allowance from this bar — see the edges={} note on SafeAreaView.
    paddingBottom: 12,
    paddingTop: 10,
    gap: 10,
    borderTopWidth: 1,
    borderTopColor: "rgba(255,255,255,0.07)",
    backgroundColor: "#000",
  },
  input: {
    flex: 1,
    backgroundColor: "#161618",
    borderRadius: 23,
    paddingHorizontal: 18,
    paddingVertical: 13,
    color: "#FFF",
    fontSize: 15,
    maxHeight: 100,
    fontFamily: FONTS.regular,
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.16)",
    // react-native-web paints the UA focus ring on a focused textarea — an
    // amber-ish outline belonging to no palette in this app. Suppressed and
    // replaced with the brand ring below rather than simply removed: an input
    // with no visible focus state is a keyboard-navigation dead end.
    ...(Platform.OS === "web" ? { outlineStyle: "none" as any } : null),
  },
  inputFocused: {
    borderColor: "rgba(216,255,42,0.55)",
  },
  sendBtn: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: COLORS.primary,
    justifyContent: "center",
    alignItems: "center",
  },
  sendBtnDisabled: { backgroundColor: "#1F1F22" },
  sendBtnTxt: { color: "#000", fontSize: 20, fontFamily: FONTS.extrabold },

  micBtn: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: "#161618",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.12)",
    justifyContent: "center",
    alignItems: "center",
  },
  micBtnActive: {
    backgroundColor: COLORS.primary,
    borderColor: COLORS.primary,
  },

  // ── Empty state ───────────────────────────────────────────────────────────────
  emptyState: { flex: 1, alignItems: "center", justifyContent: "center", paddingHorizontal: 40 },
  emptyEmoji: { fontSize: 52, marginBottom: 16 },
  emptyTitle: { color: "#FFF", fontSize: 24, fontFamily: FONTS.black, marginBottom: 8, letterSpacing: -0.5 },
  emptySub: { color: "#555555", fontSize: 15, textAlign: "center", lineHeight: 22, fontFamily: FONTS.regular },

  // ── Coach picker (no coach yet) ──
  pickerScroll: { paddingHorizontal: 20, paddingTop: 40, paddingBottom: 40, flexGrow: 1 },
  pickerIconWrap: {
    width: 56, height: 56, borderRadius: 18, alignSelf: "center",
    backgroundColor: "rgba(216,255,42,0.12)", borderWidth: 1, borderColor: "rgba(216,255,42,0.3)",
    alignItems: "center", justifyContent: "center", marginBottom: 18,
  },
  pickerTitle: { color: "#FFF", fontSize: 26, fontFamily: FONTS.black, letterSpacing: -0.6, textAlign: "center" },
  pickerSub: { color: "#666", fontSize: 14, fontFamily: FONTS.regular, lineHeight: 21, textAlign: "center", marginTop: 8, paddingHorizontal: 8 },
  pickerSectionLabel: { color: "#666", fontSize: 11.5, fontFamily: FONTS.semibold, letterSpacing: 1, marginTop: 24 },

  creatorCard: {
    flexDirection: "row", alignItems: "center", gap: 14,
    backgroundColor: "#0D0D0D", borderRadius: 20, borderWidth: 1, borderColor: "#1A1A1A", padding: 14,
  },
  creatorAvatar: { width: 56, height: 56, borderRadius: 28, backgroundColor: "#141414" },
  creatorAvatarFallback: { alignItems: "center", justifyContent: "center", borderWidth: 1, borderColor: "#222" },
  creatorAvatarInitial: { color: COLORS.primary, fontFamily: FONTS.bold, fontSize: 22 },
  creatorName: { color: "#FFF", fontFamily: FONTS.bold, fontSize: 16 },
  creatorSpecialty: { color: "#8A8A8E", fontFamily: FONTS.regular, fontSize: 12.5, marginTop: 2 },
  creatorSubs: { color: "#555", fontFamily: FONTS.regular, fontSize: 11.5, marginTop: 2 },
  creatorSelectBtn: {
    backgroundColor: COLORS.primary, borderRadius: 18, paddingHorizontal: 18, paddingVertical: 9,
  },
  creatorSelectTxt: { color: "#000", fontFamily: FONTS.bold, fontSize: 13 },

  browseAllBtn: {
    flexDirection: "row", alignItems: "center", gap: 10,
    marginTop: 22, paddingVertical: 15, paddingHorizontal: 16,
    backgroundColor: "rgba(216,255,42,0.06)", borderRadius: 16,
    borderWidth: 1, borderColor: "rgba(216,255,42,0.25)",
  },
  browseAllTxt: { color: COLORS.primary, fontFamily: FONTS.bold, fontSize: 14.5 },

  // ── Coach picker modal ────────────────────────────────────────────────────────
  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.8)", justifyContent: "flex-end" },
  modalSheet: {
    backgroundColor: "#0D0D0D",
    borderTopLeftRadius: 28,
    borderTopRightRadius: 28,
    padding: 24,
    paddingBottom: 40,
    borderTopWidth: 0.5,
    borderColor: "#1A1A1A",
  },
  modalTitle: { color: "#FFF", fontSize: 20, fontFamily: FONTS.bold, marginBottom: 4 },
  modalSub: { color: "#555555", fontSize: 13, marginBottom: 20, fontFamily: FONTS.regular },

  coachOption: {
    flexDirection: "row",
    alignItems: "center",
    gap: 14,
    backgroundColor: "#0D0D0D",
    borderRadius: 20,
    padding: 14,
    marginBottom: 10,
    borderWidth: 0.5,
    borderColor: "#1A1A1A",
  },
  coachOptionActive: {
    borderWidth: 1,
    borderColor: COLORS.primary,
  },
  coachOptionImg: { width: 46, height: 46, borderRadius: 23 },
  coachOptionImgFallback: { backgroundColor: "#2A2A2A", justifyContent: "center", alignItems: "center" },
  coachOptionName: { color: "#FFF", fontSize: 15, fontFamily: FONTS.bold },
  coachOptionSpecialty: { color: "#555555", fontSize: 12, marginTop: 2, fontFamily: FONTS.regular },
  coachOptionCheck: { color: COLORS.primary, fontSize: 18, fontFamily: FONTS.bold },
});
