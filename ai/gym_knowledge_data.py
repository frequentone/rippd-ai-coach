"""
gym_knowledge_data.py — Shared gym knowledge base for RIPPD AI coach.

Imported by:
  - rag_pipeline.py   (in-memory keyword search, zero cost)
  - ingest_gym_knowledge.py  (optional pgvector ingest if OpenAI key available)

To add more knowledge: append a new dict to GYM_KNOWLEDGE following the same
{category, tags, content} schema. Tags are the keywords that trigger retrieval —
use common user phrasings (e.g. "lat pulldown" not just "pulldown"), including
the wrong-but-common ones people actually type. Matching is substring-based and
case-insensitive, and `category` doubles as a scoring keyword (see
search_gym_knowledge), so keep categories descriptive.
"""

GYM_KNOWLEDGE: list[dict] = [
    {
        "category": "anatomy_chest",
        "tags": [
            "chest", "pec", "pecs", "pectoral", "pectoralis",
            "upper chest", "lower chest", "mid chest",
            "bench press", "incline press", "incline bench", "flat bench",
            "cable fly", "cable crossover", "dumbbell fly", "dips",
            "push", "pressing",
        ],
        "content": (
            "Pectoralis Major Anatomy & Biomechanics: Composed of three distinct regions: "
            "1. Clavicular Head (Upper Chest): Primary action is shoulder flexion and horizontal adduction. "
            "Optimized by incline presses (30-45 degree incline) and low-to-high cable flyes. "
            "2. Sternocostal Head (Mid-Chest): Primary action is horizontal adduction. "
            "Optimized by flat barbell/dumbbell bench presses and converging machine presses. "
            "3. Abdominal Head (Lower Chest): Primary action is horizontal adduction and shoulder extension from an elevated position. "
            "Optimized by high-to-low cable crossovers and dips. "
            "To maximize hypertrophy, movements must load the pectorals at long muscle lengths (deep stretch) under high mechanical tension. "
            "The clavicular head is NOT recruited by flat pressing — incline angle is mandatory to hit upper chest."
        ),
    },
    {
        "category": "anatomy_back",
        "tags": [
            "back", "lat", "lats", "latissimus", "latissimus dorsi",
            "trap", "traps", "trapezius", "rhomboid", "rhomboids", "teres",
            "row", "rows", "pull down", "pulldown", "lat pulldown", "cable row",
            "chest supported row", "seated row", "single arm row",
            "width", "thickness", "back workout",
        ],
        "content": (
            "Latissimus Dorsi & Mid-Back Biomechanics: The back is split functionally into the Latissimus Dorsi (Lats) "
            "and the Upper/Mid-Back (Trapezius, Rhomboids, Teres Major). "
            "1. Latissimus Dorsi: Fibers run vertically/diagonally. Function is shoulder adduction and shoulder extension. "
            "To isolate the lats, keep elbows tucked close to the torso during pulls (neutral grip single-arm rows or lat pulldowns) "
            "and stop the pull BEFORE scapular retraction — once the shoulder blade moves, the mid-back takes over. "
            "2. Upper/Mid-Back: Fibers run horizontally. Function is scapular retraction, elevation, and depression. "
            "To target, use wide flared elbow positioning (75-90 degrees from torso) and actively retract the shoulder blades at peak contraction "
            "(chest-supported rows, wide-grip seated rows). "
            "Key insight: elbows close to torso = lats. Elbows flared wide = mid-back."
        ),
    },
    {
        "category": "anatomy_shoulders",
        "tags": [
            "shoulder", "shoulders", "delt", "delts", "deltoid", "deltoids",
            "anterior delt", "front delt", "lateral delt", "side delt", "side raise",
            "rear delt", "posterior delt",
            "lateral raise", "side lateral raise", "cable lateral raise",
            "overhead press", "shoulder press", "military press",
            "rear delt fly", "face pull", "reverse pec deck",
            "rotator cuff",
            "wider", "width", "broad", "broader", "wide shoulders", "cap",
        ],
        "content": (
            "Deltoid Complex & Rotator Cuff Mechanics: The shoulder comprises three heads: "
            "1. Anterior Deltoid: Responsible for shoulder flexion. Heavily taxed during all compound pressing (bench press, overhead press) — "
            "almost always already overtrained in a typical push program. Isolated via front raises, which are rarely needed. "
            "2. Lateral Deltoid: Responsible for shoulder abduction, creates the wide-shoulder look. "
            "It operates on a poor mechanical advantage and requires isolation — compound pressing does NOT hit it. "
            "Cable lateral raises (behind the back or Y-raises) are superior to dumbbell laterals because cables maintain "
            "continuous tension through the full range; dumbbells drop to near-zero load at the bottom of the movement. "
            "3. Posterior Deltoid: Responsible for horizontal abduction and external rotation. Severely undertrained in most programs. "
            "Best isolated via rear delt cable flyes, face pulls, or reverse pec deck with a neutral/pronated grip and elbows flared at 90 degrees. "
            "Priority order: rear delts and lateral delts need the most direct work — front delts rarely need isolation."
        ),
    },
    {
        "category": "anatomy_arms_triceps",
        "tags": [
            "tricep", "triceps", "tricep workout", "arm",
            "long head", "lateral head", "medial head",
            "overhead extension", "skull crusher", "tricep pushdown",
            "close grip bench", "cable extension", "rope pushdown",
            "tate press", "kickback", "tricep dip",
            "elbow extension",
        ],
        "content": (
            "Triceps Brachii Biomechanics: The triceps consist of three heads: Long, Lateral, and Medial. "
            "The Long Head is a biarticular muscle crossing both the elbow and shoulder joints — it drives elbow extension AND shoulder extension. "
            "CRITICAL: the long head is only fully stretched when the arm is raised overhead. "
            "To achieve maximal mechanical tension on the long head, overhead extensions are REQUIRED "
            "(cable overhead extensions, EZ-bar overhead tricep extension, dumbbell overhead extension). "
            "Pushdowns do NOT stretch the long head and therefore only partially train the tricep. "
            "The Lateral and Medial heads only cross the elbow joint and are maximally taxed via pushdowns "
            "(V-bar or straight bar) and heavy close-grip pressing where the shoulder position is neutral. "
            "NEVER use dumbbell kickbacks — the resistance curve is terrible (near zero load at peak contraction). "
            "Optimal tricep program: overhead extension (long head) + pushdown (lateral/medial head) + close-grip bench (mass)."
        ),
    },
    {
        "category": "anatomy_arms_biceps",
        "tags": [
            "bicep", "biceps", "curl", "curls", "arm",
            "long head", "short head", "brachialis", "brachioradialis",
            "hammer curl", "hammer curls", "incline curl", "preacher curl",
            "reverse curl", "supination", "forearm", "elbow flexion",
            "concentration curl", "spider curl", "cable curl",
        ],
        "content": (
            "Biceps Brachii & Brachialis Biomechanics: The anterior upper arm contains the Biceps Brachii "
            "(Long Head/Outer, Short Head/Inner) and the underlying Brachialis. "
            "1. Biceps Brachii: Biarticular muscle responsible for elbow flexion AND forearm supination. "
            "Long Head is placed under a greater stretch when the shoulder is extended (elbow behind the torso) → "
            "Incline Dumbbell Curls, Cable Curls with elbow pulled back. "
            "Short Head is emphasized when elbow is in front of the body → Preacher Curls, concentration curls. "
            "SUPINATION matters: rotating the pinky up at the top of the curl maximally contracts the bicep. "
            "2. Brachialis & Brachioradialis: The brachialis sits UNDER the bicep and pushes it up — "
            "growing brachialis makes the bicep look bigger even without growing the bicep itself. "
            "To target brachialis, remove supination advantage by using neutral grip (Hammer Curls) "
            "or pronated grip (Reverse Curls). "
            "Optimal bicep program: incline curl or cable curl (long head stretch) + preacher/spider curl (short head peak) + hammer curl (brachialis)."
        ),
    },
    {
        "category": "anatomy_legs_quads",
        "tags": [
            "quad", "quads", "quadriceps", "leg", "legs",
            "squat", "squats", "hack squat", "leg press", "leg extension",
            "rectus femoris", "vastus", "knee extension",
            "knee dominant", "front squat", "Bulgarian split squat",
            "sissy squat", "lunges",
        ],
        "content": (
            "Quadriceps Biomechanics: Composed of four distinct muscles: Rectus Femoris, Vastus Lateralis, "
            "Vastus Medialis (VMO), and Vastus Intermedius. "
            "The three Vastus muscles are monoarticular (cross only the knee joint) → they are maximally loaded "
            "during deep knee flexion: Hack Squats, Leg Presses, Barbell Squats. "
            "IMPORTANT: To maximize quad activation, the knee must travel far forward over the toes (high tibial angle). "
            "A vertical shin kills quad recruitment and shifts load to the glutes. "
            "The Rectus Femoris is biarticular, crossing both the hip and knee. "
            "CRITICAL FLAW of compound squats: during a squat, the hip flexion shortens the rectus femoris "
            "while the knee extension lengthens it — these cancel out, making it difficult to grow via squatting alone. "
            "To isolate and grow the rectus femoris, open-chain knee extensions (Leg Extensions) or Sissy Squats are required. "
            "Optimal quad program: deep hack squat or leg press (vastus muscles) + leg extension (rectus femoris)."
        ),
    },
    {
        "category": "anatomy_legs_hamstrings_glutes",
        "tags": [
            "hamstring", "hamstrings", "glute", "glutes", "gluteus",
            "rdl", "romanian deadlift", "stiff leg", "hip hinge",
            "leg curl", "seated leg curl", "prone leg curl", "lying leg curl",
            "hip thrust", "glute bridge",
            "Bulgarian split squat", "deficit lunge",
            "posterior chain", "hip extension", "knee flexion",
        ],
        "content": (
            "Hamstrings & Glutes Biomechanics: "
            "1. Hamstrings: Composed of the Biceps Femoris (long + short heads), Semitendinosus, and Semimembranosus. "
            "All except the short head of biceps femoris are biarticular, crossing hip AND knee → hip extension + knee flexion. "
            "For maximal hypertrophy, BOTH functions must be loaded: "
            "Seated Leg Curls are superior to prone/lying leg curls — the seated position keeps the hip flexed, "
            "which LENGTHENS the hamstrings at the hip simultaneously, creating peak tension at a long muscle length. "
            "Romanian Deadlifts / Stiff-Legged Deadlifts load the hamstrings through hip extension at long muscle lengths — "
            "feel the stretch at the bottom, NOT the squeeze at the top. "
            "Optimal hamstring program: seated leg curl (knee flexion at length) + RDL (hip extension load). "
            "2. Gluteus Maximus: The primary hip extensor. Peak glute activation requires full hip extension at the top "
            "AND a loaded deep hip flexion at the bottom. "
            "Hip Thrusts provide the highest glute EMG because the hip is extended against load at the top. "
            "Deficit Bulgarian Split Squats provide a deep hip stretch under load. "
            "RDLs hit glutes secondarily but are not optimal for peak glute mass — pair with hip thrusts."
        ),
    },
    {
        "category": "anatomy_legs_calves",
        "tags": [
            "calf", "calves", "calf raise", "standing calf raise", "seated calf raise",
            "gastrocnemius", "soleus",
            "leg press calf raise", "donkey calf raise",
            "ankle", "plantarflexion",
        ],
        "content": (
            "Calf Complex Biomechanics: Composed of the Gastrocnemius and the Soleus. "
            "Gastrocnemius is biarticular — crosses both the knee and ankle. Drives plantarflexion AND assists knee flexion. "
            "CRITICAL: When the knee is bent (seated calf raise), the gastrocnemius enters active insufficiency "
            "(it's already shortened at the knee end and cannot produce force at the ankle end). "
            "This shifts nearly 100% of the load to the Soleus. "
            "To target the highly visible Gastrocnemius: calf raises MUST be done with a STRAIGHT, LOCKED knee "
            "(Standing Calf Raises, Calf Raises on a leg press with legs straight). "
            "To target the Soleus (the deeper muscle that adds thickness to the lower leg): "
            "use Seated Calf Raises or calf raises with bent knee. "
            "Optimal calf program: standing calf raises (gastrocnemius) + seated calf raises (soleus). "
            "Both heads require full range of motion — sink into the deepest stretch possible at the bottom."
        ),
    },
    {
        "category": "anatomy_core",
        "tags": [
            "core", "abs", "ab", "abdominal", "abdominals",
            "rectus abdominis", "oblique", "obliques", "transverse abdominis",
            "crunch", "sit up", "leg raise", "cable crunch",
            "plank", "hollow body", "ab wheel", "decline crunch",
            "six pack", "lower abs", "upper abs",
        ],
        "content": (
            "Core & Abdominal Biomechanics: "
            "1. Rectus Abdominis: Runs vertically from sternum to pubis — responsible for spinal flexion. "
            "There is no 'upper abs' vs 'lower abs' muscle — it is one muscle, but the neural signal differs. "
            "Upper region (sternum toward pelvis): Crunches, Cable Crunches, Decline Crunches. "
            "Lower region (pelvis toward sternum, harder to activate): Leg Raises, Hanging Knee Raises, Reverse Crunches. "
            "CRITICAL: Sit-ups are inferior to crunches — the hip flexors do most of the work after the first 30 degrees. "
            "Cable crunches are superior to bodyweight crunches — they load the abs at long muscle lengths with constant tension. "
            "2. Obliques (Internal + External): Responsible for lateral flexion and rotation. "
            "Best targeted with rotational exercises: wood chops, cable rotations, side bends. "
            "3. Transverse Abdominis: Deep stabilizer — trained via bracing (Valsalva maneuver during heavy compound lifts), "
            "planks, and hollow body holds. Cannot be seen but is critical for spinal stability and injury prevention. "
            "Six-pack visibility is almost entirely determined by body fat percentage, not ab exercise volume."
        ),
    },
    {
        "category": "hypertrophy_physiology",
        "tags": [
            "hypertrophy", "muscle growth", "muscle building",
            "sets", "reps", "volume", "frequency",
            "rir", "rpe", "failure", "to failure",
            "mechanical tension", "progressive overload",
            "how many sets", "how many reps", "optimal",
            "mtor", "muscle protein synthesis", "recovery",
            "split", "push pull legs", "ppl",
        ],
        "content": (
            "Hypertrophy Training Principles & Intensity Metrics: Muscle growth is primarily governed by Mechanical Tension, "
            "which triggers the mTORC1 signaling pathway. "
            "1. Volume: The optimal dose-response baseline is 10 to 20 working sets per muscle group per week. "
            "Below 10 sets = under-stimulated. Above 20 sets = junk volume with diminishing returns and excess fatigue. "
            "2. Proximity to Failure: Working sets must finish within 0 to 2 Reps in Reserve (RIR), "
            "translating to a Rate of Perceived Exertion (RPE) of 8 to 10. "
            "Training completely to absolute mechanical failure (0 RIR) maximizes motor unit recruitment but incurs "
            "high CNS fatigue — use on isolation movements only, not heavy compound lifts. "
            "3. Rep Range: Hypertrophy is uniform across a 6 to 30 rep spectrum IF sets are matched for RIR. "
            "6-12 reps is most time-efficient for minimizing systemic fatigue while maximizing tension. "
            "4. Progressive Overload: Must increase total volume (weight × reps × sets) over time for continued growth. "
            "Add reps before adding weight. Once you can do 3 sets of 12 at a given weight, increase the load. "
            "5. Frequency: 2× per muscle per week is optimal for most natural lifters — allows sufficient volume "
            "while giving 48-72 hours of recovery between sessions."
        ),
    },
    {
        "category": "nutrition_recovery",
        "tags": [
            "protein", "protein intake", "how much protein",
            "calories", "calorie", "caloric", "surplus", "deficit", "bulk", "cut", "recomp",
            "macros", "macro", "carbs", "fat", "fiber",
            "muscle protein synthesis", "leucine",
            "deload", "recovery", "rest", "overtraining",
            "water", "hydration",
            "maintenance calories", "tdee",
        ],
        "content": (
            "Nutrition, Macronutrient Architecture & Recovery Thermodynamics: "
            "1. Energy Balance: "
            "For lean muscle gain: 250 to 500 kcal surplus above maintenance (TDEE). Too large a surplus = unnecessary fat gain. "
            "For fat loss with muscle preservation: 300 to 500 kcal deficit. Larger deficits accelerate muscle loss. "
            "2. Protein Intake: Must be sustained at 1.6 to 2.2 grams per kilogram of total body weight daily. "
            "Individual meals should contain 0.4-0.55g/kg of protein to trigger the leucine threshold (~3g leucine), "
            "which activates muscle protein synthesis machinery. "
            "Spreading protein across 3-5 meals is superior to 1-2 large doses. "
            "3. Carbohydrates: Primary fuel for glycolytic training. Do not slash carbs when training hard — "
            "performance and muscle retention both suffer. Time carbs around training for best fuel and recovery. "
            "4. Fats: Minimum 0.5-1g per kg of bodyweight for hormonal health (testosterone production). "
            "Do not drop fat too low on a cut. "
            "5. Deload: A deload week (reducing total volume by 40-50%, keeping intensity at RPE 6-7) is required every "
            "6 to 8 weeks of hard training to shed accumulated CNS and connective tissue fatigue. "
            "Connective tissue (tendons/ligaments) repairs significantly slower than muscle — "
            "ignoring this causes overuse injuries."
        ),
    },
    {
        "category": "programming_principles",
        "tags": [
            "program", "programming", "workout plan", "split",
            "push pull legs", "ppl", "upper lower", "full body", "bro split",
            "periodization", "linear progression", "strength",
            "compound", "isolation", "exercise order", "superset",
            "warm up", "warmup", "cool down",
            "beginner", "intermediate", "advanced",
        ],
        "content": (
            "Training Program Design Principles: "
            "1. Exercise Order: Always perform compound movements (bench, squat, deadlift, row, press) FIRST "
            "when the nervous system is fresh. Isolation exercises (curls, laterals, extensions) come after. "
            "Never pre-fatigue a prime mover before a compound lift unless deliberately planned. "
            "2. Splits by level: "
            "Beginner (0-1yr): Full body 3x/week. Frequency beats volume — hit every muscle 3x/week with 6-8 compound movements. "
            "Intermediate (1-3yr): Upper/Lower 4x/week or Push/Pull/Legs 6x/week. "
            "Advanced (3yr+): PPL or specialization blocks, 12-20 sets per muscle per week. "
            "3. Compound-to-Isolation ratio: 60-70% compound, 30-40% isolation. "
            "All isolation, no compound = slow strength gains. All compound, no isolation = lagging smaller muscles. "
            "4. Supersets: Pairing antagonist muscles (biceps/triceps, chest/back) reduces rest time and improves training density "
            "without compromising performance. Pairing agonists (two quad exercises back to back) significantly reduces performance. "
            "5. Warm-up: 2-3 progressive warm-up sets at 40-60-80% of working weight before the first heavy set. "
            "Skipping warm-ups on heavy compound lifts is the fastest way to get injured."
        ),
    },
    {
        "category": "time_efficient_training",
        "tags": [
            "no time", "short on time", "busy", "short workout", "quick workout",
            "time efficient", "how long should i train", "session length",
            "rest period", "rest time", "rest between sets", "how long to rest",
            "superset", "supersets", "drop set", "drop sets", "dropset",
            "rest pause", "rest-pause", "advanced technique",
            "minimum sets", "how many sets do i need", "maintenance volume",
            "maintain muscle", "detraining", "missed a week", "took a break",
            "general warm up", "do i need to warm up", "stretching before workout",
        ],
        "content": (
            "Time-Efficient Training & Rest Period Science (Iversen et al. 2021, Sports Medicine): "
            "1. Minimum effective dose: at least 4 working sets per muscle group per week produces real strength "
            "and hypertrophy gains — this is the floor for someone with very limited time, not the target. "
            "Above 4, gains scale up to roughly 10+ weekly sets before returns flatten; add volume only if the "
            "time cost is worth it. "
            "2. Load range: 6-15RM is the default working range for both strength and hypertrophy. If a user is "
            "training at home with light weights (15-40+ reps), they need to push those sets close to failure — "
            "light loads only work as well as heavy loads when effort is equated. "
            "3. Frequency vs volume: weekly volume matters more than how it's split up. Training a muscle once a "
            "week with all sets performed, or spread across 3 sessions, produces similar results if total sets "
            "are equal — so a user who can only train once a week is not necessarily leaving gains on the table, "
            "as long as they hit enough sets that session. "
            "4. Rest periods: untrained lifters can use 1-2 min rest between sets without losing strength gains; "
            "trained lifters should use 2+ min, especially on heavy compound lifts, to sustain training volume load. "
            "Shorter rest is fine for small isolation movements. "
            "5. Advanced time-saving techniques — supersets (agonist/antagonist pairing), drop sets, and rest-pause "
            "training — roughly HALVE session time at equal training volume. They are more effective for hypertrophy "
            "than for maximal strength. Avoid them on heavy free-weight compound lifts (squat, deadlift, bench) for "
            "safety — reserve them for machines and isolation exercises. "
            "6. Warm-up: a general cardio warm-up gives little to no benefit for strength training performance. "
            "Only a specific warm-up (light-load ramp-up sets of the actual lift) matters, and mainly for heavy loads "
            "(>80% 1RM). Static stretching before lifting acutely reduces strength output — skip it pre-workout; "
            "save flexibility work for after training or a separate session. "
            "7. Maintenance: after a hard training block, strength and muscle mass can be maintained for weeks with "
            "as little as one session per week at reduced volume — reassure users returning from a break or a busy "
            "week that they haven't lost their gains, they just need to resume consistent training."
        ),
    },
    {
        "category": "lifting_technique_evidence",
        "tags": [
            "correct form", "proper form", "lifting technique", "how to lift",
            "neutral spine", "flat back", "rounded back", "back rounding",
            "lift with your legs", "squat lift", "stoop lift", "hip hinge technique",
            "is rounding my back bad", "spine flexion", "bending my back",
            "deadlift form", "safe lifting", "form check", "technique",
        ],
        "content": (
            "Lifting Technique — What the Evidence Actually Supports (Washmuth et al. 2022, IJSPT): "
            "There is no single universally 'correct' lifting technique — the old rule 'always squat-lift, never round "
            "your back' is not well supported by evidence. Key points for coaching: "
            "1. Squat-style lifting (bent knees, upright torso) has NOT been shown to reduce low-back injury risk "
            "compared to a more hip-hinge/stoop-style lift (bent torso, straighter knees) in the research literature. "
            "2. A truly 'neutral spine' under load is largely a coaching cue, not a biomechanical reality — even "
            "well-cued lifters show measurable lumbar flexion under load. Some flexion during a lift is normal and "
            "not automatically dangerous; injury risk is about mismatch between load and an individual's current "
            "tissue tolerance, not about hitting one specific joint angle. "
            "3. Technique should match the task and the person: stoop-style lifting is more metabolically efficient "
            "and faster for high-rep/high-volume work (e.g. high-rep deadlifts, conditioning work), while squat-style "
            "lifting reduces knee flexion demand and may suit someone with knee/patellofemoral pain. "
            "4. For a user in acute low-back pain, don't force a single 'ideal' form — let pain guide which technique "
            "variant feels least provocative right now, and reintroduce variability as pain resolves. "
            "5. Practical coaching takeaway: cue lifters toward bracing, control, and a load appropriate to their "
            "current capacity rather than rigidly enforcing one universal form for every person and every lift."
        ),
    },
    {
        "category": "anabolic_window_timing",
        "tags": [
            "anabolic window", "post workout protein", "post workout meal",
            "when should i eat after workout", "post workout timing",
            "protein timing", "do i need to eat right after training",
            "meal timing", "how soon after workout should i eat",
            "pre workout meal", "does timing matter",
        ],
        "content": (
            "Protein Timing & the 'Anabolic Window': total daily protein intake and overall caloric intake are the "
            "highest-signal predictors of muscle preservation and hypertrophy — not the timing of any single meal. "
            "The idea of a hyper-urgent 30-minute post-workout window is a myth; if a balanced meal was eaten within "
            "4-5 hours before training, there is already a circulating pool of amino acids available and no rush to "
            "eat immediately after. Post-workout nutrition is still a good, convenient opportunity to hit daily "
            "protein and calorie targets, and there's no upside to delaying a meal past 1-2 hours out of habit — but "
            "the priority is spreading total daily protein across 3-5 meals in a way that fits the person's "
            "lifestyle and adherence, not hitting a rigid post-workout clock."
        ),
    },
    {
        "category": "physique_proportion",
        "tags": [
            "look bigger", "bigger", "look better", "aesthetic", "aesthetics",
            "proportion", "proportions", "proportionate", "v-taper", "v taper",
            "vtaper", "frame", "look wider", "look broad", "broad", "wider",
            "skinny", "tiny", "small", "short", "height", "look taller",
            "athletic", "physique", "chopped", "stocky", "blocky", "bulky",
            "how do i look bigger", "where do i start", "want to look",
        ],
        "content": (
            "Looking Bigger & More Aesthetic (Proportion Over Size): For anyone who looks skinny or small — "
            "regardless of height — the goal is not maximum body weight, it is the V-taper: broad shoulders and "
            "back tapering to a lean waist. This is what makes a frame read as athletic and imposing, and it makes "
            "shorter lifters look taller and larger, not blocky. The visual levers, in priority order: "
            "1. Lateral deltoids — the single biggest lever for shoulder width and the 'wide' look (cable/dumbbell "
            "lateral raises; compound presses do NOT build them). "
            "2. Upper back and lats — width and thickness (pull-ups, lat pulldowns, rows). "
            "3. Upper chest — incline pressing for a fuller, higher chest line. "
            "4. Arms — visible mass that fills a sleeve. "
            "5. A relatively lean waist — staying under ~15-18% body fat keeps the taper sharp; excess fat widens "
            "the waist and kills the V. "
            "Short/stocky 'chopped' looks come from bulking aggressively with no attention to proportion, not from "
            "height itself. Build the shoulders and back, keep the waist lean, and the frame looks bigger without "
            "gaining excessive weight."
        ),
    },
    {
        "category": "beginner_getting_started",
        "tags": [
            "beginner", "beginners", "new to gym", "just started", "start",
            "getting started", "where do i start", "how do i start", "first time",
            "never worked out", "newbie", "starting out", "should i go to the gym",
            "how to begin", "complete beginner", "what should i do",
        ],
        "content": (
            "Beginner Getting Started Principles: A new lifter's fastest visible progress comes from a simple, "
            "consistent full-body or upper/lower routine built on compound movements, not from complex specialization. "
            "Core priorities: 1. Train each muscle group 2x/week with compounds (squat, hinge, horizontal + vertical "
            "press, horizontal + vertical pull). 2. Progressive overload — add a small amount of weight or a rep "
            "whenever a set is completed with good form; this is the actual driver of growth. 3. Learn form on lighter "
            "loads first; leave 1-2 reps in reserve rather than grinding to failure early on. 4. Consistency and "
            "recovery (7-9h sleep, enough protein ~1.6-2.2g/kg bodyweight) beat program-hopping. Beginners can build "
            "muscle and lose fat at the same time ('newbie gains'), so the exact surplus/deficit matters less than "
            "showing up, progressing the weights, and hitting protein. Start with 3-4 sessions/week and let adherence, "
            "not intensity, be the goal for the first few months."
        ),
    },
    {
        "category": "soreness_doms",
        "tags": [
            "sore", "soreness", "doms", "muscle soreness", "aching", "ache",
            "stiff", "tight", "day after", "next day", "can barely walk",
            "not sore", "no soreness", "wasn't sore", "is soreness good",
            "should i train sore", "training sore", "recovery",
        ],
        "content": (
            "DOMS (Delayed Onset Muscle Soreness): Soreness appearing 12-72h after training, peaking around 24-48h. "
            "It is caused by mechanical stress and the inflammatory/repair response — NOT by lactic acid, which clears "
            "within roughly an hour of a session. "
            "CRITICAL: soreness is NOT a measure of workout quality and NOT required for growth. It mostly reflects "
            "NOVELTY (a new exercise, more volume, or lots of lengthened/eccentric work). A well-adapted lifter making "
            "excellent progress is often barely sore — this is the 'repeated bout effect', where the same session "
            "produces far less soreness the second time. Chasing soreness leads to program-hopping and junk volume. "
            "Training a sore muscle is safe and does not impair growth as long as force output and technique are intact. "
            "Light movement and blood flow reduce soreness faster than passive rest; static stretching does not prevent "
            "or meaningfully cure it. Sleep and adequate protein are the real levers. "
            "DISTINGUISH FROM INJURY: DOMS is dull, diffuse, in the muscle belly, symmetrical, tender to press, and eases "
            "as you warm up. Pain that is sharp, in or around a JOINT, one-sided, present at rest, worsening session over "
            "session, or accompanied by swelling, numbness, or tingling is NOT DOMS and needs assessment, not a workout."
        ),
    },
    {
        "category": "protein_requirements_distribution",
        "tags": [
            "protein", "how much protein", "protein intake", "protein target",
            "protein per day", "grams of protein", "protein timing",
            "protein per meal", "whey", "protein shake", "protein powder",
            "leucine", "muscle protein synthesis", "mps", "too much protein",
        ],
        "content": (
            "Protein Requirements & Distribution: For building or retaining muscle, total daily protein is the dominant "
            "variable. Meta-analytic evidence (Morton et al. 2018) shows benefits plateau around 1.6g per kg bodyweight "
            "per day, with ~2.2g/kg as a practical ceiling that covers individual variation. Eating beyond this is not "
            "harmful, just not more effective. "
            "IN A CALORIE DEFICIT, protein needs go UP, not down — 2.0-2.4g/kg protects lean mass while dieting, because "
            "the body is more willing to break down muscle for energy. Protein is also the most satiating macronutrient "
            "and has the highest thermic effect (20-30% of its calories are burned digesting it), so it does double duty "
            "on a cut. "
            "DISTRIBUTION: spread intake across 3-5 meals of roughly 0.4g/kg each. Each feeding needs about 2.5-3g of "
            "leucine to maximally trigger muscle protein synthesis — that's roughly 20-40g of a quality protein, or a "
            "larger serving of a plant source. One large protein meal a day does not match the same total spread out. "
            "Total daily intake still matters far more than exact timing. "
            "SAFETY: high protein does not damage healthy kidneys — this is a persistent myth with no support in people "
            "with normal kidney function. Anyone with existing kidney disease should follow medical advice instead."
        ),
    },
    {
        "category": "vegetarian_protein_india",
        "tags": [
            "vegetarian", "veg", "vegan", "plant protein", "no meat", "no egg",
            "eggless", "paneer", "dal", "lentils", "chana", "chickpea", "rajma",
            "soya", "soy", "tofu", "curd", "dahi", "yogurt", "greek yogurt",
            "milk", "sattu", "peanut", "besan", "sprouts", "tempeh", "quinoa",
            "vegetarian protein", "protein without meat", "indian protein",
        ],
        "content": (
            "Vegetarian & Indian Protein Strategy: Hitting 1.6-2.2g/kg on a vegetarian Indian diet is very achievable, "
            "but it fails by DEFAULT because a typical thali is carb-dominant — the protein has to be added deliberately "
            "to every meal, not hoped for. "
            "HIGH-YIELD SOURCES (protein per 100g, uncooked unless noted): soya chunks ~52g, whey isolate ~85g, "
            "paneer ~18-20g, greek yogurt/hung curd ~9-10g, tofu ~8g, moong/masoor/chana dal ~24g dry (~7-9g cooked), "
            "rajma/chole ~22g dry, besan ~22g, peanuts ~25g, sattu ~20g, eggs ~13g (~6g each). "
            "THE TWO REAL PROBLEMS: "
            "1. LEUCINE. Plant proteins are lower in leucine, so a plant meal needs to be roughly 30-50% larger than a "
            "dairy/whey equivalent to trigger the same muscle protein synthesis. Soy, paneer, dairy and whey are the "
            "highest-leucine vegetarian options — anchor meals around them. "
            "2. CALORIE COST. Dal and rajma carry a lot of carbs per gram of protein. Getting 40g of protein from dal "
            "alone drags in far more calories than 40g from paneer, soya chunks, curd, or whey — this is why "
            "vegetarians on a cut stall. "
            "PRACTICAL FIXES: combine cereals with pulses (roti+dal, rice+rajma) so the amino acid profile completes; "
            "swap regular curd for hung/greek curd; use soya chunks in pulao and curries; add a scoop of whey or sattu "
            "to hit the daily total. Vegetarians also tend to have LOWER baseline creatine stores, so creatine "
            "supplementation often produces a noticeably bigger response than it does in meat eaters."
        ),
    },
    {
        "category": "rep_ranges_load_goals",
        "tags": [
            "rep range", "reps", "how many reps", "how many sets", "sets and reps",
            "heavy or light", "high reps", "low reps", "8-12", "5x5", "3x10",
            "strength vs size", "hypertrophy reps", "strength reps", "toning",
            "tone", "lean out", "1rm", "percentage", "load",
        ],
        "content": (
            "Rep Ranges, Load & Goal: Hypertrophy is remarkably rep-range agnostic. When sets are taken close to failure, "
            "growth is similar across roughly 5-30 reps (Schoenfeld et al.) — the muscle responds to high mechanical "
            "tension and effort, not to a magic number. The 8-12 'hypertrophy zone' is popular because it's efficient "
            "(enough load, tolerable fatigue, sane set duration), not because it's uniquely anabolic. "
            "STRENGTH IS DIFFERENT and is NOT rep-range agnostic. Maximal strength is highly specific to heavy load and "
            "practising the skill of the lift — it requires work at roughly 80%+ of 1RM, generally 1-6 reps, with long "
            "rest. You cannot build a maximal squat with sets of 20. "
            "PRACTICAL: build strength with 3-6 reps on the main compounds; build size with 6-15 on compounds and 10-20 "
            "on isolation where joint stress is lower and the loading is easier to manage. "
            "'TONING' IS NOT A TRAINING STYLE. There is no such thing as a rep range that lengthens, sculpts, or tones a "
            "muscle. The 'toned' look is muscle that already exists becoming visible as body fat drops. High reps with "
            "light weight do not burn fat off the muscle being trained — spot reduction is not physiologically possible. "
            "Anyone chasing 'tone' needs the same plan as anyone chasing size: build/retain muscle, and lose fat via a "
            "calorie deficit."
        ),
    },
    {
        "category": "proximity_to_failure_rir",
        "tags": [
            "failure", "to failure", "rir", "reps in reserve", "rpe",
            "how hard", "intensity", "leave reps", "grinding", "junk volume",
            "am i training hard enough", "effort", "push to failure",
        ],
        "content": (
            "Proximity to Failure (RIR): RIR = Reps In Reserve — how many more reps you could have done. 0 RIR is "
            "failure; 2 RIR means you had two left. "
            "Effort is the trigger for growth, but sets taken to roughly 0-4 RIR produce SIMILAR hypertrophy — the last "
            "rep before failure is not magic. What failure reliably adds is fatigue: it disproportionately raises "
            "recovery cost, cuts the quality of your following sets, and on heavy compounds it degrades technique and "
            "raises injury risk for no extra growth. "
            "PRACTICAL DEFAULT: 1-3 RIR on compound lifts. Taking isolation work (curls, lateral raises, leg extensions) "
            "to or near failure is low-risk and fine, because the fatigue cost is local and technique breakdown is minor. "
            "THE COMMON FAILURE MODE runs the other way: most people stop far short of a genuinely hard set and think "
            "they're at failure. Untrained lifters routinely misjudge by 4-6 reps. If the bar speed hasn't slowed at all "
            "on the last rep, the set was too easy — that's junk volume: work that costs time and recovery but is not "
            "close enough to failure to signal growth."
        ),
    },
    {
        "category": "training_volume_frequency",
        "tags": [
            "volume", "how many sets", "sets per week", "weekly volume",
            "frequency", "how often", "how many times a week", "twice a week",
            "more sets", "overtraining", "too much", "junk volume", "split",
        ],
        "content": (
            "Training Volume & Frequency: Volume — counted as hard sets per muscle per week — is the primary driver of "
            "hypertrophy once effort and progression are in place. "
            "LANDMARKS: roughly 10-20 hard sets per muscle per week is the effective range for most trainees. Under ~6 "
            "sets is maintenance-level for a trained lifter. Beginners grow well on the lower end (8-12) and should not "
            "start at the ceiling — you want the SMALLEST dose that still progresses, so there's somewhere to go later. "
            "Above ~20 sets returns diminish sharply and recovery becomes the limiter, not the stimulus. "
            "FREQUENCY: when weekly volume is equated, training a muscle 2x/week beats 1x/week (Schoenfeld et al. 2016). "
            "The mechanism is simple — muscle protein synthesis after a session is elevated for roughly 24-48h, so two "
            "stimuli per week keeps it elevated more of the time. Beyond 2-3x/week, equated volume shows little further "
            "benefit; frequency is mainly a tool for SPLITTING volume so each session's sets stay high quality. "
            "This is the real argument against the classic bro split: hitting chest once a week means 5 sessions of "
            "growth signal per muscle per month instead of 8-10. "
            "OVERTRAINING (true clinical overtraining) is rare in recreational lifters. What people actually hit is "
            "under-recovery: too much volume relative to sleep, food, and life stress. The symptoms are stalled or "
            "falling performance, persistent joint ache, poor sleep, and lost motivation — address recovery and volume, "
            "not willpower."
        ),
    },
    {
        "category": "deload_fatigue_management",
        "tags": [
            "deload", "deloading", "rest week", "back off week", "plateau",
            "stalled", "stuck", "not progressing", "no progress", "burnt out",
            "burned out", "fatigue", "tired all the time", "joints hurt",
            "strength dropped", "getting weaker", "regression",
        ],
        "content": (
            "Deloads & Fatigue Management: Fatigue accumulates faster than it feels. A deload is a planned reduction "
            "that lets accumulated fatigue clear so the fitness you already built can show up as performance. "
            "WHEN: either scheduled every 4-8 weeks of hard training, or autoregulated — triggered by the actual signals. "
            "The signals: performance flat or falling for 2+ consecutive sessions on lifts that were progressing, "
            "persistent joint ache, elevated resting heart rate, disrupted sleep, or motivation dropping off a cliff. "
            "Advanced lifters and anyone in a calorie deficit need them more often; a true beginner rarely needs a "
            "deload as early as they think, because they cannot yet generate enough fatigue to warrant one. "
            "HOW: keep training and keep the movements — cut VOLUME by roughly half (drop sets, not exercises) while "
            "keeping load near normal, OR keep sets and drop intensity ~10-20%. Cutting load AND volume to zero for a "
            "week isn't a deload, it's detraining, and it feels worse coming back. Deloads run about one week. "
            "A STALL IS NOT AUTOMATICALLY A DELOAD PROBLEM. Check the cheaper explanations first, in order: are they "
            "eating enough (a stalled bench on a deficit is a nutrition problem), sleeping enough, actually training "
            "close to failure, and progressing the load at all? Most 'plateaus' are one of those four, not fatigue."
        ),
    },
    {
        "category": "sleep_recovery",
        "tags": [
            "sleep", "sleeping", "rest", "recovery", "tired", "insomnia",
            "how much sleep", "8 hours", "nap", "late night", "night shift",
            "not sleeping", "sleep quality", "growth hormone",
        ],
        "content": (
            "Sleep & Recovery: Sleep is the single highest-leverage recovery variable and the one most often traded away. "
            "TARGET: 7-9 hours. Chronic restriction below ~6h measurably reduces performance, raises perceived exertion "
            "(the same weight feels heavier), impairs glucose handling, raises appetite via increased ghrelin and reduced "
            "leptin, and blunts muscle protein synthesis. "
            "THE CRITICAL FINDING FOR ANYONE DIETING: in a calorie deficit, sleep restriction changes WHAT you lose, not "
            "just how you feel. Nedeltcheva et al. (2010) put dieters on 5.5h vs 8.5h of sleep with identical calories — "
            "the sleep-deprived group lost roughly 55% LESS fat and 60% MORE lean mass for the same weight on the scale. "
            "Cutting sleep to fit in a workout is usually a net loss. "
            "PRACTICAL LEVERS, in order of impact: consistent sleep and wake times (including weekends), a dark cool "
            "room, no caffeine within 6-8h of bed (its half-life is ~5h — a 4pm coffee is still half in your system at "
            "9pm), and limiting alcohol, which reliably wrecks sleep architecture even when it helps you fall asleep. "
            "Naps of 20-30 min genuinely help recover a short night. "
            "Growth hormone does spike during deep sleep, but the muscle-building benefit of sleep comes from the whole "
            "recovery picture, not from chasing a GH spike."
        ),
    },
    {
        "category": "supplements_evidence",
        "tags": [
            "supplement", "supplements", "creatine", "whey", "bcaa", "bcaas",
            "glutamine", "pre workout", "preworkout", "fat burner", "mass gainer",
            "testosterone booster", "test booster", "should i take", "worth it",
            "multivitamin", "omega 3", "fish oil", "ashwagandha",
        ],
        "content": (
            "Supplements — What Actually Has Evidence: The supplement industry sells almost entirely on hope. Only a "
            "short list survives scrutiny. "
            "CREATINE MONOHYDRATE — the most researched sports supplement in existence and the only one that reliably "
            "adds muscle and strength. 3-5g per day, EVERY day, timing irrelevant. Loading (20g/day for 5-7 days) only "
            "makes stores saturate faster; skipping it gets you to the same place in ~3-4 weeks. Monohydrate is the "
            "studied form — HCl, buffered, and 'advanced' versions cost more for no proven advantage. Expect ~1-2kg of "
            "early scale weight from intracellular water; this is inside the muscle, not bloat, and is not fat. It is "
            "safe long-term in healthy people, does not cause hair loss or kidney damage, and does not require cycling. "
            "Vegetarians typically respond MORE strongly, having lower baseline stores. "
            "PROTEIN POWDER — not magic, just convenient food. Useful only if you're short of your daily protein target. "
            "CAFFEINE — genuinely improves performance (see the caffeine entry). "
            "VITAMIN D — worth it only if deficient, which is common in people with little sun exposure. "
            "WHAT TO SKIP: BCAAs (pointless if you're hitting your daily protein — they're an incomplete subset of what "
            "whole protein already gives you), glutamine, testosterone boosters, fat burners (caffeine plus marketing), "
            "and mass gainers (sugar you could get cheaper from actual food). "
            "No supplement compensates for insufficient protein, calories, sleep, or training that isn't hard enough."
        ),
    },
    {
        "category": "caffeine_preworkout",
        "tags": [
            "caffeine", "coffee", "pre workout", "preworkout", "energy drink",
            "black coffee", "tea", "before workout", "energy", "stimulant",
            "how much caffeine", "tolerance",
        ],
        "content": (
            "Caffeine for Training: One of very few ergogenic aids with strong, consistent evidence. It improves "
            "endurance, reduces perceived exertion (the set feels easier at the same load), and modestly improves "
            "strength and power output. "
            "DOSE: 3-6mg per kg bodyweight, 30-60 minutes before training. For a 70kg lifter that's roughly 200-400mg — "
            "about 2-3 cups of coffee. More is not better; above ~6mg/kg you mostly buy jitters, elevated heart rate, "
            "and anxiety without extra performance. "
            "TOLERANCE: builds within 1-4 weeks of daily use. Escalating the dose to chase the original effect is how "
            "people end up at 600mg with sleep problems. Better: keep the dose moderate and save the higher end for hard "
            "sessions, or take a 1-2 week reset if it has stopped doing anything. "
            "THE SLEEP TRADE-OFF IS THE REAL COST. Caffeine's half-life is about 5 hours, so half of a 4pm dose is still "
            "circulating at 9pm. Caffeine for an evening session can quietly cost the sleep that drives your recovery — "
            "a bad trade. Avoid within 6-8h of bed; for late trainers, either use a much smaller dose or skip it. "
            "Pre-workout blends are mostly caffeine plus beta-alanine (which causes the harmless tingling and does very "
            "little for a lifter) — plain coffee is cheaper and does the same job."
        ),
    },
    {
        "category": "cardio_interference",
        "tags": [
            "cardio", "running", "run", "treadmill", "hiit", "steps", "walking",
            "interference", "cardio kill gains", "cardio and weights",
            "before or after", "should i do cardio", "conditioning", "cycling",
        ],
        "content": (
            "Cardio & the Interference Effect: The fear that cardio 'kills gains' is mostly overblown, but the effect is "
            "real at the extremes. "
            "WHAT THE EVIDENCE SAYS: interference scales with the frequency, duration, and intensity of endurance work. "
            "Moderate cardio (2-4 sessions/week) has minimal impact on hypertrophy and improves work capacity, recovery "
            "between sets, and heart health. High-volume, high-intensity endurance training — especially running, which "
            "adds a large eccentric load to the legs — does meaningfully blunt strength and power adaptations. "
            "MINIMISING IT: (1) separate cardio and lifting by 6+ hours, or put them on different days; (2) if they must "
            "share a session and muscle/strength is the goal, LIFT FIRST — do the thing you care about while you're "
            "fresh; (3) prefer low-impact modalities (cycling, incline walking, rowing) over running for leg recovery; "
            "(4) don't hammer intense cardio the day before a heavy leg session. "
            "FOR FAT LOSS: cardio is a tool for burning calories, not a requirement. The deficit is what causes fat loss. "
            "Daily step count (NEAT) is usually a bigger and far more sustainable lever than HIIT — steps cost almost no "
            "recovery, whereas HIIT competes directly with lifting for the same recovery budget. Aim for 8-10k steps "
            "before adding formal cardio. Never drop lifting for cardio while cutting: resistance training is what tells "
            "the body to keep the muscle."
        ),
    },
    {
        "category": "fat_loss_rate_expectations",
        "tags": [
            "lose fat", "fat loss", "how fast", "how long", "deficit", "cutting",
            "cut", "lose weight", "how much weight", "per week", "crash diet",
            "1kg a week", "realistic", "belly fat", "spot reduction", "abs",
            "love handles", "target fat",
        ],
        "content": (
            "Fat Loss — Rate & Expectations: Fat loss requires a sustained calorie deficit. Nothing else on this list "
            "matters if that isn't in place. "
            "RATE: 0.5-1% of bodyweight per week is the sustainable range. For an 80kg person that's 0.4-0.8kg/week. "
            "Leaner people should sit at the low end; people with more fat to lose can tolerate the high end. Pushing "
            "faster increasingly costs LEAN MASS, not just fat, and tanks training performance, sleep, and adherence — "
            "which is why crash diets rebound. A ~20-25% deficit from maintenance is the usual sweet spot. "
            "PROTECT MUSCLE while cutting with two non-negotiables: keep lifting heavy (this is the signal to retain "
            "muscle) and keep protein high (2.0-2.4g/kg). Cutting weights and living on cardio is how people arrive at "
            "goal weight still soft — 'skinny fat' is a muscle problem, not a fat problem. "
            "SPOT REDUCTION IS NOT POSSIBLE. No exercise burns fat from the area it trains. Crunches do not remove belly "
            "fat; inner-thigh machines do not slim thighs. Fat is mobilised systemically and the ORDER is largely "
            "genetic — the stubborn area (lower belly, hips, love handles for most) is simply the last to go. Abs become "
            "visible at a low enough body fat, roughly 10-12% for men and 18-22% for women; there is no ab exercise that "
            "shortcuts that. "
            "WEIGHT FLUCTUATES 1-2kg daily from water, sodium, carbs, and gut contents. Judge progress on a weekly "
            "average and on measurements/photos, never on a single morning's reading."
        ),
    },
    {
        "category": "muscle_gain_rate_expectations",
        "tags": [
            "build muscle", "muscle gain", "how fast", "how long", "bulk",
            "bulking", "surplus", "gain weight", "newbie gains", "how much muscle",
            "realistic", "dirty bulk", "lean bulk", "recomp", "body recomposition",
            "gain muscle lose fat", "at the same time",
        ],
        "content": (
            "Muscle Gain — Rate & Expectations: Muscle is built far more slowly than the internet implies, and "
            "unrealistic expectations are the main reason people quit or dirty bulk. "
            "REALISTIC RATE (roughly, for men; women should expect about half in absolute kg, though similar as a "
            "percentage of their starting mass): Year 1 ~0.9-1.1kg/month. Year 2 ~0.5kg/month. Year 3 ~0.2-0.3kg/month. "
            "Year 4+ ~0.1kg/month or less. Rate is driven by training age, not calendar age — an untrained 40-year-old "
            "still gets newbie gains. "
            "THE SURPLUS SHOULD BE SMALL. A ~10% surplus (roughly 200-400 kcal/day) supports the maximum rate muscle can "
            "actually be built at. Eating 1000 over won't build muscle faster — the ceiling is biological, not caloric — "
            "it just adds fat, which then costs a cut to remove. Aim for ~0.25-0.5% bodyweight gained per month. "
            "RECOMPOSITION (building muscle and losing fat at once) is genuinely possible for: beginners, people "
            "returning after a layoff (muscle memory), people with higher body fat, and anyone whose protein or training "
            "was previously poor. It gets progressively harder as you get leaner and more trained. Expect the SCALE TO "
            "BARELY MOVE during a recomp — that is the plan working, not stalling. Track strength, measurements and "
            "photos instead of weight. "
            "NEVER JUDGE MUSCLE GAIN ON THE SCALE WEEK TO WEEK. Early weight jumps are glycogen, water and creatine, "
            "not muscle."
        ),
    },
    {
        "category": "fat_loss_plateau",
        "tags": [
            "plateau", "stalled", "stuck", "not losing", "weight not moving",
            "scale not moving", "stopped losing", "metabolism", "starvation mode",
            "damaged metabolism", "eating less not working", "diet break",
            "refeed", "adaptation",
        ],
        "content": (
            "Fat Loss Plateaus: A genuine plateau means no change in weekly average weight or measurements for 2-3+ "
            "weeks. Anything shorter is normal fluctuation and needs no action. "
            "WHY THEY HAPPEN, in order of actual likelihood: "
            "1. ADHERENCE DRIFT (by far the most common). Intake creeps up — untracked oil and ghee, weekend meals, "
            "liquid calories, 'bites and tastes', portion estimates sliding. Studies consistently show people "
            "underreport intake by 20-50%, and they are not lying — they genuinely don't see it. Check this FIRST, "
            "always, before touching calories. "
            "2. NEAT DOWN-REGULATION. As you diet, non-exercise activity drops unconsciously — you fidget less, walk "
            "less, take the lift. This can quietly account for hundreds of calories a day. Tracking steps catches it. "
            "3. LOWER TDEE FROM LOWER BODYWEIGHT. A lighter body simply burns less. The deficit that worked at 90kg is "
            "smaller at 82kg — the target has to be recalculated, this is just arithmetic. "
            "4. ADAPTIVE THERMOGENESIS — a real but modest effect (roughly 10-15% below predicted), NOT the mythical "
            "'starvation mode'. Nobody stops losing fat in a real deficit; metabolism does not 'break'. "
            "FIXES, in order: tighten tracking for two weeks before changing anything; raise daily steps; only then cut "
            "calories by ~5-10%; or take a DIET BREAK (2 weeks at maintenance) which restores hormones, training "
            "performance, and adherence. Do NOT respond to every stall by cutting calories again — that road ends at an "
            "unsustainably low intake with nowhere left to go."
        ),
    },
    {
        "category": "home_minimal_equipment",
        "tags": [
            "home", "home workout", "no gym", "without gym", "no equipment",
            "bodyweight", "dumbbells only", "resistance band", "bands",
            "small gym", "limited equipment", "travel", "hotel", "no machines",
            "no barbell", "can i build muscle at home",
        ],
        "content": (
            "Home & Minimal-Equipment Training: You can absolutely build muscle without a full gym. Hypertrophy needs "
            "mechanical tension, effort close to failure, and progressive overload — a barbell is one way to deliver "
            "that, not the only way. Evidence shows bodyweight and band training build muscle comparably when effort "
            "and progression are matched; the real constraint is that it gets HARDER TO PROGRESSIVELY OVERLOAD once the "
            "load stops going up. "
            "HOW TO PROGRESS WITHOUT ADDING WEIGHT, in order of usefulness: add reps; slow the eccentric (3-4s down); "
            "increase range of motion (deficit push-ups, elevated-foot split squats); move to a harder leverage "
            "(incline → flat → decline → archer → one-arm push-up); reduce rest between sets; add pauses at the "
            "lengthened position; progress to unilateral (single-leg/single-arm) versions, which effectively double the "
            "load per limb. "
            "MOVEMENT SUBSTITUTIONS: Bench press → push-up progressions, floor press with dumbbells. Squat → goblet "
            "squat, split squat, Bulgarian split squat, step-up. Deadlift/hinge → single-leg RDL, hip thrust, glute "
            "bridge, band good morning. Row → inverted row under a table, band row, single-arm dumbbell row. Pulldown → "
            "pull-up/chin-up, band pulldown. Overhead press → pike push-up, band overhead press. "
            "THE HARDEST GAP is the posterior chain and heavy pulling — a pull-up bar and one set of adjustable "
            "dumbbells or a few bands closes most of it for very little money. Legs are the other gap: bodyweight leg "
            "work must go unilateral and high-rep to be hard enough to matter."
        ),
    },
    {
        "category": "women_training_evidence",
        "tags": [
            "women", "woman", "female", "girl", "girls", "ladies",
            "bulky", "get bulky", "too big", "toned", "tone up",
            "lift heavy", "period", "menstrual", "cycle", "pregnancy",
            "women lifting", "should women lift",
        ],
        "content": (
            "Training for Women — What the Evidence Says: The training principles do not change by sex. Women build "
            "muscle at a similar RELATIVE rate to men (similar percentage gains), just from a smaller starting mass, so "
            "absolute kg gained is lower. The programming answer is the same: compounds, progressive overload, enough "
            "protein. "
            "'LIFTING MAKES WOMEN BULKY' IS FALSE. Women have roughly 10-20x less circulating testosterone, which makes "
            "accidental large-scale muscle gain physiologically implausible — the muscle women say they want ('toned') "
            "IS the muscle lifting builds, revealed as body fat drops. Nobody gets bulky by accident; it takes years of "
            "deliberate effort even for men. "
            "WOMEN ACTUALLY RECOVER SOMEWHAT BETTER between sets and sessions and tend to tolerate higher volume and "
            "higher reps than men at the same relative load — largely due to greater fatigue resistance. This means "
            "shorter rest periods are often workable and there is no reason to avoid heavy work. "
            "MENSTRUAL CYCLE: current evidence does NOT support rigid cycle-based programming — the studies are mixed "
            "and effects are small and highly individual. Do not prescribe a cycle-phased plan as if it were settled "
            "science. Autoregulate instead: train as normal, and on days where symptoms genuinely reduce performance, "
            "reduce load or volume rather than skipping. Iron deficiency is meaningfully more common in menstruating "
            "women, especially vegetarians, and shows up as unexplained fatigue and dropping performance — that one is "
            "worth a blood test rather than a program change."
        ),
    },
    {
        "category": "alcohol_hydration",
        "tags": [
            "alcohol", "drink", "drinking", "beer", "wine", "whiskey", "booze",
            "cheat", "party", "weekend", "hangover",
            "water", "hydration", "dehydrated", "how much water", "electrolytes",
        ],
        "content": (
            "Alcohol & Hydration: "
            "ALCOHOL costs a lifter on four fronts, and the calories are the least of them. (1) It acutely blunts muscle "
            "protein synthesis — heavy intake after training reduces the adaptation from the session you just paid for, "
            "even when protein is adequate. (2) It reliably degrades sleep architecture, suppressing REM, so recovery "
            "drops even if total hours look fine. (3) It carries 7 kcal/g with zero nutritional value and, while "
            "alcohol is being metabolised, fat oxidation is suppressed because the body prioritises clearing it. "
            "(4) It kills next-day adherence — the drinking usually isn't what breaks the diet, the food around it is. "
            "The practical line: 1-2 drinks occasionally is not going to undo months of work; regular heavy drinking "
            "will quietly cap your progress. If you're drinking, keep it away from your post-training window and "
            "protect the next day's sleep. "
            "HYDRATION: aim for roughly 30-40ml per kg bodyweight per day as a baseline, plus replacement for training "
            "losses. As little as 2% dehydration measurably reduces strength, endurance, and concentration — and thirst "
            "lags behind actual need, so it's a poor gauge. Urine colour is the practical check: pale straw is the "
            "target, dark yellow means catch up. Electrolytes only matter for long sessions, heavy sweaters, or hot "
            "climates — for a normal hour in an air-conditioned gym, water is enough. Don't overdo it either: drinking "
            "extreme volumes doesn't help and can dilute sodium."
        ),
    },
]
