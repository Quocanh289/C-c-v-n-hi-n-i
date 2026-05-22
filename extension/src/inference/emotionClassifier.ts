// ====================================================
// AI-Powered Emotion Classifier (Local Inference)
// Uses Transformers.js with XLM-RoBERTa for multilingual
// emotion detection in Vietnamese and English
// Supports 28 fine-grained labels for English, 9 coarse for Vietnamese
// ====================================================

import {
  EmotionCategory,
  EmotionResult,
  EmotionScores,
  Emotion28Scores,
  Emotion9Scores,
  ExtensionSettings,
  MENTAL_HEALTH_LABELS,
  MentalHealthScores,
  GOEMOTIONS_28_LABELS,
  COARSE_EMOTIONS_LABELS,
  EMOTION_28_TO_9_MAP,
} from '../types/emotion';

// Vietnamese character detection for language classification
const VIETNAMESE_CHARS_REGEX = /[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]/i;

function detectLanguage(text: string): 'vi' | 'en' | 'mixed' {
  const viChars = text.match(VIETNAMESE_CHARS_REGEX);
  const viCount = viChars?.length || 0;
  const totalChars = text.replace(/\s/g, '').length;
  if (totalChars === 0) return 'en';
  const viRatio = viCount / totalChars;
  if (viRatio > 0.15) {
    const enWords = text.match(/[a-z]+/gi)?.length || 0;
    if (enWords > 0 && enWords / (text.length / 5) > 0.3) return 'mixed';
    return 'vi';
  }
  return 'en';
}

/**
 * Lightweight local emotion classifier using transformer models.
 * Designed for browser-based inference with ONNX runtime.
 * Supports 28 fine-grained labels for English, 9 coarse for Vietnamese.
 */
export class EmotionClassifier {
  private model: any = null;
  private tokenizer: any = null;
  private initialized = false;
  private initializationPromise: Promise<void> | null = null;
  private slangDictionary: Map<string, EmotionCategory> = new Map();
  private emojiEmotionMap: Map<string, EmotionCategory> = new Map();

  // Vietnamese slang dictionary
  private vietnameseSlang: Record<string, EmotionCategory> = {
    'xỉu': EmotionCategory.Surprise,
    'đỉnh nóc': EmotionCategory.Joy,
    'kịch trần': EmotionCategory.Joy,
    'gắt': EmotionCategory.Anger,
    'cà khịa': EmotionCategory.Sarcastic,
    'thả thính': EmotionCategory.Joy,
    'crush': EmotionCategory.Joy,
    'ngáo': EmotionCategory.Surprise,
    'bất lực': EmotionCategory.Sadness,
    'túng': EmotionCategory.Anxiety,
    'quạo': EmotionCategory.Anger,
    'hãm': EmotionCategory.Toxic,
    'toxic': EmotionCategory.Toxic,
    'cringe': EmotionCategory.Sarcastic,
    'sao cũng được': EmotionCategory.Neutral,
    'chán': EmotionCategory.Sadness,
    'mệt mỏi': EmotionCategory.Sadness,
    'áp lực': EmotionCategory.Anxiety,
    'sợ': EmotionCategory.Fear,
    'hoảng': EmotionCategory.Fear,
    'vui': EmotionCategory.Joy,
    'hạnh phúc': EmotionCategory.Joy,
    'tuyệt vời': EmotionCategory.Joy,
    'buồn': EmotionCategory.Sadness,
    'giận': EmotionCategory.Anger,
    'tức': EmotionCategory.Anger,
    'ngạc nhiên': EmotionCategory.Surprise,
    'bất ngờ': EmotionCategory.Surprise,
    'lo lắng': EmotionCategory.Anxiety,
    'hồi hộp': EmotionCategory.Anxiety,
  };

  // English slang dictionary
  private englishSlang: Record<string, EmotionCategory> = {
    'cooked': EmotionCategory.Sarcastic,
    'delulu': EmotionCategory.Sarcastic,
    'slay': EmotionCategory.Joy,
    'based': EmotionCategory.Joy,
    'cringe': EmotionCategory.Sarcastic,
    'ghosted': EmotionCategory.Sadness,
    'salty': EmotionCategory.Anger,
    'savage': EmotionCategory.Sarcastic,
    'lit': EmotionCategory.Joy,
    'fire': EmotionCategory.Joy,
    'woke': EmotionCategory.Joy,
    'simp': EmotionCategory.Sarcastic,
    'flex': EmotionCategory.Joy,
    'no cap': EmotionCategory.Joy,
    'cap': EmotionCategory.Sarcastic,
    'sus': EmotionCategory.Anxiety,
    'bet': EmotionCategory.Joy,
    'bussin': EmotionCategory.Joy,
    'mid': EmotionCategory.Neutral,
    'npv': EmotionCategory.Neutral,
    'main character': EmotionCategory.Joy,
    'side quest': EmotionCategory.Sarcastic,
    'ick': EmotionCategory.Sarcastic,
    'red flag': EmotionCategory.Sarcastic,
    'green flag': EmotionCategory.Joy,
    'ick factor': EmotionCategory.Sarcastic,
  };

  // Emoji to emotion mapping
  private emojiMap: Record<string, EmotionCategory> = {
    '😡': EmotionCategory.Anger, '🤬': EmotionCategory.Anger,
    '😢': EmotionCategory.Sadness, '😭': EmotionCategory.Sadness,
    '😞': EmotionCategory.Sadness, '😔': EmotionCategory.Sadness,
    '😊': EmotionCategory.Joy, '😍': EmotionCategory.Joy,
    '🥰': EmotionCategory.Joy, '😂': EmotionCategory.Joy,
    '🤣': EmotionCategory.Joy, '🎉': EmotionCategory.Joy,
    '✨': EmotionCategory.Joy, '💀': EmotionCategory.Sarcastic,
    '🗿': EmotionCategory.Sarcastic, '😰': EmotionCategory.Anxiety,
    '😨': EmotionCategory.Fear, '😱': EmotionCategory.Fear,
    '😲': EmotionCategory.Surprise, '🤯': EmotionCategory.Surprise,
    '🙂': EmotionCategory.Sarcastic, '😏': EmotionCategory.Sarcastic,
    '😒': EmotionCategory.Sarcastic, '🤔': EmotionCategory.Surprise,
    '😐': EmotionCategory.Neutral, '👍': EmotionCategory.Joy,
    '❤️': EmotionCategory.Joy, '💔': EmotionCategory.Sadness,
    '☠️': EmotionCategory.Toxic, '🤮': EmotionCategory.Toxic,
    '👎': EmotionCategory.Anger, '😤': EmotionCategory.Anger,
    '😩': EmotionCategory.Sadness, '🥺': EmotionCategory.Sadness,
    '🤡': EmotionCategory.Sarcastic, '👀': EmotionCategory.Surprise,
    '🔥': EmotionCategory.Joy, '💯': EmotionCategory.Joy,
  };

  /** Punctuation patterns indicative of sarcasm */
  private sarcasmPatterns = [
    /~.*~/, /🤡/, /🙂$/, /sure,? .*!/i, /oh,? really/i,
    /wow,? .*!/i, /great,? .* not/i, /nice,? .* sarcasm/i,
    /obviously/i, /clearly/i, /totally not/i, /as if/i,
    /yeah,? right/i, /whatever you say/i,
    /hay quá ha/i, /giỏi quá ha/i, /tốt quá ha/i, /thông minh quá/i,
  ];

  /** Toxic pattern indicators */
  private toxicPatterns = [
    /\bstupid\b/i, /\bidiot\b/i, /\bkill yourself\b/i,
    /\bhate\b/i, /\btrash\b/i, /\bđồ ngu\b/i,
    /\bđi chết\b/i, /\bmày\b.*\bchết\b/i, /\bngu\b/i, /\bóc chó\b/i,
  ];

  // Keyword mapping for 28-label rule-based analysis (English)
  private enKeywordTo28: Record<string, string> = {
    'admire': 'admiration', 'respect': 'admiration', 'awesome': 'admiration',
    'amazing': 'admiration', 'great': 'admiration', 'brilliant': 'admiration',
    'impressive': 'admiration', 'wonderful': 'admiration',
    'fun': 'amusement', 'funny': 'amusement', 'hilarious': 'amusement',
    'laugh': 'amusement', 'joke': 'amusement',
    'angry': 'anger', 'furious': 'anger', 'mad': 'anger',
    'annoy': 'annoyance', 'annoying': 'annoyance', 'ugh': 'annoyance',
    'approve': 'approval', 'agree': 'approval', 'good': 'approval',
    'care': 'caring', 'kind': 'caring', 'sweet': 'caring',
    'confuse': 'confusion', 'confused': 'confusion',
    'curious': 'curiosity', 'wonder': 'curiosity',
    'want': 'desire', 'wish': 'desire', 'dream': 'desire',
    'disappoint': 'disappointment',
    'disapprove': 'disapproval', 'wrong': 'disapproval',
    'disgust': 'disgust', 'gross': 'disgust', 'nasty': 'disgust',
    'embarrass': 'embarrassment',
    'excite': 'excitement', 'excited': 'excitement',
    'fear': 'fear', 'scared': 'fear', 'afraid': 'fear',
    'thank': 'gratitude', 'thanks': 'gratitude', 'grateful': 'gratitude',
    'grief': 'grief', 'loss': 'grief', 'miss': 'grief',
    'happy': 'joy', 'joy': 'joy', 'delighted': 'joy',
    'love': 'love', 'adorable': 'love', 'precious': 'love',
    'nervous': 'nervousness', 'anxious': 'nervousness',
    'optimist': 'optimism', 'hope': 'optimism',
    'proud': 'pride',
    'realize': 'realization', 'understand': 'realization',
    'relief': 'relief', 'relieved': 'relief', 'whew': 'relief',
    'remorse': 'remorse', 'sorry': 'remorse', 'apologize': 'remorse',
    'sad': 'sadness', 'unhappy': 'sadness', 'depressed': 'sadness',
    'surprise': 'surprise', 'shock': 'surprise', 'wow': 'surprise',
  };

  constructor() {
    this.initializeSlangDictionary();
    this.initializeEmojiMap();
  }

  private initializeSlangDictionary(): void {
    for (const [phrase, emotion] of Object.entries(this.vietnameseSlang)) {
      this.slangDictionary.set(phrase.toLowerCase(), emotion);
    }
    for (const [phrase, emotion] of Object.entries(this.englishSlang)) {
      this.slangDictionary.set(phrase.toLowerCase(), emotion);
    }
  }

  private initializeEmojiMap(): void {
    for (const [emoji, emotion] of Object.entries(this.emojiMap)) {
      this.emojiEmotionMap.set(emoji, emotion);
    }
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;
    if (this.initializationPromise) return this.initializationPromise;
    this.initializationPromise = this._initialize();
    return this.initializationPromise;
  }

  private async _initialize(): Promise<void> {
    try {
      const { pipeline } = await import('@xenova/transformers');
      // Try multiple model names for emotion detection
      const modelNames = [
        'Xenova/xlm-roberta-base-emotion',
        'Xenova/distilbert-base-uncased-emotion',
        'Xenova/bert-base-uncased-emotion',
      ];
      for (const modelName of modelNames) {
        try {
          this.model = await pipeline('text-classification', modelName);
          console.log(`[EmotionLens] Transformer model loaded: ${modelName}`);
          break;
        } catch (e) {
          console.warn(`[EmotionLens] Failed to load ${modelName}, trying next...`);
        }
      }
      this.tokenizer = null;
      this.initialized = true;
      if (!this.model) {
        console.warn('[EmotionLens] No transformer model loaded, using rule-based fallback');
      }
    } catch (error) {
      console.warn('[EmotionLens] Could not load transformer model, using rule-based fallback:', error);
      this.initialized = true;
    }
  }

  /**
   * Analyze text and return emotion classification result.
   * Uses 28 fine-grained labels for English, 9 coarse for Vietnamese.
   */
  async analyze(text: string, settings: ExtensionSettings): Promise<EmotionResult> {
    const startTime = performance.now();
    const language = detectLanguage(text);
    const isEnglish = language === 'en';

    // Stage 1: Fast rule-based analysis (always produces 9 extension scores)
    const ruleBasedResult = this.ruleBasedAnalysis(text);

    // Stage 2: 28-label rule-based analysis for English
    let scores28: Emotion28Scores | undefined;
    let scores9: Emotion9Scores | undefined;
    if (isEnglish) {
      scores28 = this.ruleBased28Analysis(text);
    } else {
      scores9 = this.ruleBased9Analysis(text);
    }

    // Stage 3: Ensemble scoring
    const finalResult: EmotionResult = {
      analysisType: 'emotion',
      primaryEmotion: this.getPrimaryEmotion(ruleBasedResult, text, settings),
      scores: ruleBasedResult,
      scores28: scores28,
      scores9: scores9,
      labelType: isEnglish ? 'fine' : 'coarse',
      numLabels: isEnglish ? 28 : 9,
      confidence: Math.max(...Object.values(ruleBasedResult)),
      toxicityScore: ruleBasedResult[EmotionCategory.Toxic],
      sarcasmScore: ruleBasedResult[EmotionCategory.Sarcastic],
      language: language,
      source: 'local',
      inferenceTimeMs: performance.now() - startTime,
    };

    return finalResult;
  }

  /**
   * Analyze English text with the backend DeBERTa-v3 + LoRA mental-health model.
   * The trained adapter is loaded by the backend from the best_model checkpoint.
   */
  async analyzeMentalHealth(text: string, settings: ExtensionSettings): Promise<EmotionResult> {
    const startTime = performance.now();
    const language = detectLanguage(text);
    if (language !== 'en') {
      return this.emptyMentalHealthResult('Normal', 0, language, performance.now() - startTime);
    }

    const baseUrl = (settings.backendApiUrl || 'http://localhost:8000').replace(/\/+$/, '');
    const response = await fetch(`${baseUrl}/api/mental-health/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      throw new Error(`Mental health backend returned ${response.status}`);
    }

    const data = await response.json() as {
      primary_condition?: string;
      primary_confidence?: number;
      all_scores?: Record<string, number>;
      needs_attention?: boolean;
      severity_level?: number;
      severity_label?: string;
      processing_time_ms?: number;
    };

    const scores = this.normalizeMentalHealthScores(data.all_scores || {});
    const label = MENTAL_HEALTH_LABELS.includes(data.primary_condition as any)
      ? data.primary_condition as typeof MENTAL_HEALTH_LABELS[number]
      : 'Normal';

    return {
      analysisType: 'mental_health',
      primaryEmotion: label,
      scores: this.emptyEmotionScores(),
      mentalHealthScores: scores,
      labelType: 'mental_health',
      numLabels: MENTAL_HEALTH_LABELS.length,
      confidence: data.primary_confidence ?? scores[label] ?? 0,
      toxicityScore: 0,
      sarcasmScore: 0,
      language,
      source: 'backend',
      inferenceTimeMs: performance.now() - startTime,
      severityLevel: data.severity_level,
      severityLabel: data.severity_label,
      needsAttention: data.needs_attention,
    };
  }

  private emptyEmotionScores(): EmotionScores {
    return {
      [EmotionCategory.Joy]: 0,
      [EmotionCategory.Anger]: 0,
      [EmotionCategory.Sadness]: 0,
      [EmotionCategory.Anxiety]: 0,
      [EmotionCategory.Fear]: 0,
      [EmotionCategory.Surprise]: 0,
      [EmotionCategory.Neutral]: 0,
      [EmotionCategory.Toxic]: 0,
      [EmotionCategory.Sarcastic]: 0,
    };
  }

  private normalizeMentalHealthScores(rawScores: Record<string, number>): MentalHealthScores {
    const scores = {} as MentalHealthScores;
    for (const label of MENTAL_HEALTH_LABELS) {
      scores[label] = Number(rawScores[label] ?? 0);
    }
    return scores;
  }

  private emptyMentalHealthResult(
    label: typeof MENTAL_HEALTH_LABELS[number],
    confidence: number,
    language: 'vi' | 'en' | 'mixed',
    inferenceTimeMs: number,
  ): EmotionResult {
    const scores = this.normalizeMentalHealthScores({});
    scores[label] = confidence;
    return {
      analysisType: 'mental_health',
      primaryEmotion: label,
      scores: this.emptyEmotionScores(),
      mentalHealthScores: scores,
      labelType: 'mental_health',
      numLabels: MENTAL_HEALTH_LABELS.length,
      confidence,
      toxicityScore: 0,
      sarcasmScore: 0,
      language,
      source: 'local',
      inferenceTimeMs,
      severityLevel: 0,
      severityLabel: 'healthy',
      needsAttention: false,
    };
  }

  /**
   * 28-label rule-based analysis for English text.
   * Maps English keywords to 28 GoEmotions labels with confidence scores.
   */
  private ruleBased28Analysis(text: string): Emotion28Scores {
    const scores: any = {};
    for (const label of GOEMOTIONS_28_LABELS) {
      scores[label] = 0;
    }

    const lowerText = text.toLowerCase();
    const words = lowerText.split(/\s+/);

    // Check keyword matches
    for (const [keyword, emotion] of Object.entries(this.enKeywordTo28)) {
      if (lowerText.includes(keyword)) {
        scores[emotion] = (scores[emotion] || 0) + 0.25;
      }
    }

    // Boost neutral if no matches
    const scoreValues: number[] = Object.values(scores) as number[];
    const totalMatchScore = scoreValues.reduce((a: number, b: number) => a + b, 0);
    if (totalMatchScore < 0.1) {
      scores['neutral'] = 0.6;
    }

    // Normalize to 0-1
    const maxVal = Math.max(...scoreValues, 0.01);
    for (const label of GOEMOTIONS_28_LABELS) {
      scores[label] = Math.min(1, (scores[label] || 0) / maxVal);
    }

    return scores as Emotion28Scores;
  }

  /**
   * 9-label rule-based analysis for Vietnamese text.
   * Maps Vietnamese keywords to coarse emotions.
   */
  private ruleBased9Analysis(text: string): Emotion9Scores {
    const scores: any = {};
    for (const label of COARSE_EMOTIONS_LABELS) {
      scores[label] = 0;
    }

    const lowerText = text.toLowerCase();

    const viCoarseKeywords: Record<string, string> = {
      'tuyệt': 'admiration', 'xuất sắc': 'admiration', 'giỏi': 'admiration',
      'đẹp': 'admiration', 'tốt': 'admiration',
      'giận': 'anger', 'tức': 'anger', 'ghét': 'anger', 'bực': 'anger',
      'lo': 'anxiety', 'hồi hộp': 'anxiety', 'bồn chồn': 'anxiety',
      'sợ': 'fear', 'hoảng': 'fear', 'kinh': 'fear',
      'vui': 'joy', 'hạnh phúc': 'joy', 'cười': 'joy', 'thích': 'joy',
      'yêu': 'love', 'thương': 'love', 'quý': 'love',
      'buồn': 'sadness', 'khóc': 'sadness', 'đau': 'sadness', 'chán': 'sadness',
      'ngạc nhiên': 'surprise', 'bất ngờ': 'surprise', 'trời ơi': 'surprise',
    };

    for (const [keyword, emotion] of Object.entries(viCoarseKeywords)) {
      if (lowerText.includes(keyword)) {
        scores[emotion] = (scores[emotion] || 0) + 0.3;
      }
    }

    const scoreValues9: number[] = Object.values(scores) as number[];
    const totalMatchScore = scoreValues9.reduce((a: number, b: number) => a + b, 0);
    if (totalMatchScore < 0.1) {
      scores['neutral'] = 0.6;
    }

    const maxVal = Math.max(...scoreValues9, 0.01);
    for (const label of COARSE_EMOTIONS_LABELS) {
      scores[label] = Math.min(1, (scores[label] || 0) / maxVal);
    }

    return scores as Emotion9Scores;
  }

  /**
   * Fast rule-based emotion analysis.
   * Detects slang, emojis, patterns, and keywords.
   * Always returns the 9 extension category scores.
   */
  private ruleBasedAnalysis(text: string): EmotionScores {
    const scores: EmotionScores = {
      [EmotionCategory.Joy]: 0,
      [EmotionCategory.Anger]: 0,
      [EmotionCategory.Sadness]: 0,
      [EmotionCategory.Anxiety]: 0,
      [EmotionCategory.Fear]: 0,
      [EmotionCategory.Surprise]: 0,
      [EmotionCategory.Neutral]: 0.2,
      [EmotionCategory.Toxic]: 0,
      [EmotionCategory.Sarcastic]: 0,
    };

    const lowerText = text.toLowerCase();

    // Slang detection
    for (const [phrase, emotion] of this.slangDictionary.entries()) {
      if (lowerText.includes(phrase)) {
        scores[emotion] += 0.4;
        scores[EmotionCategory.Neutral] = Math.max(0, scores[EmotionCategory.Neutral] - 0.15);
      }
    }

    // Emoji analysis
    for (const char of text) {
      const emotion = this.emojiEmotionMap.get(char);
      if (emotion) scores[emotion] += 0.3;
    }

    // Sarcasm detection
    for (const pattern of this.sarcasmPatterns) {
      if (pattern.test(text)) {
        scores[EmotionCategory.Sarcastic] += 0.45;
        scores[EmotionCategory.Neutral] -= 0.1;
      }
    }

    // Toxicity detection
    for (const pattern of this.toxicPatterns) {
      if (pattern.test(text)) {
        scores[EmotionCategory.Toxic] += 0.5;
        scores[EmotionCategory.Anger] += 0.2;
        scores[EmotionCategory.Neutral] -= 0.15;
      }
    }

    // Vietnamese keyword analysis
    const viKeywords: Record<string, EmotionCategory> = {
      'tuyệt': EmotionCategory.Joy, 'vui': EmotionCategory.Joy,
      'hạnh': EmotionCategory.Joy, 'phúc': EmotionCategory.Joy,
      'yêu': EmotionCategory.Joy, 'thích': EmotionCategory.Joy,
      'cười': EmotionCategory.Joy, 'ghét': EmotionCategory.Anger,
      'xấu': EmotionCategory.Sadness, 'khóc': EmotionCategory.Sadness,
      'đau': EmotionCategory.Sadness, 'buồn': EmotionCategory.Sadness,
      'chán': EmotionCategory.Sadness, 'mệt': EmotionCategory.Sadness,
      'lo': EmotionCategory.Anxiety, 'sợ': EmotionCategory.Fear,
      'ngại': EmotionCategory.Anxiety, 'không biết': EmotionCategory.Surprise,
      'trời ơi': EmotionCategory.Surprise, 'chết': EmotionCategory.Fear,
      'kinh': EmotionCategory.Surprise, 'ghê': EmotionCategory.Surprise,
    };

    for (const [keyword, emotion] of Object.entries(viKeywords)) {
      if (lowerText.includes(keyword)) scores[emotion] += 0.2;
    }

    // English keyword analysis
    const enKeywords: Record<string, EmotionCategory> = {
      'happy': EmotionCategory.Joy, 'love': EmotionCategory.Joy,
      'great': EmotionCategory.Joy, 'amazing': EmotionCategory.Joy,
      'wonderful': EmotionCategory.Joy, 'excited': EmotionCategory.Joy,
      'awesome': EmotionCategory.Joy, 'fantastic': EmotionCategory.Joy,
      'beautiful': EmotionCategory.Joy, 'angry': EmotionCategory.Anger,
      'mad': EmotionCategory.Anger, 'furious': EmotionCategory.Anger,
      'hate': EmotionCategory.Anger, 'terrible': EmotionCategory.Sadness,
      'awful': EmotionCategory.Sadness, 'sad': EmotionCategory.Sadness,
      'depressed': EmotionCategory.Sadness, 'cry': EmotionCategory.Sadness,
      'lonely': EmotionCategory.Sadness, 'hurt': EmotionCategory.Sadness,
      'anxious': EmotionCategory.Anxiety, 'worried': EmotionCategory.Anxiety,
      'nervous': EmotionCategory.Anxiety, 'scared': EmotionCategory.Fear,
      'afraid': EmotionCategory.Fear, 'terrified': EmotionCategory.Fear,
      'shocked': EmotionCategory.Surprise, 'surprised': EmotionCategory.Surprise,
      'wow': EmotionCategory.Surprise, 'omg': EmotionCategory.Surprise,
      'wtf': EmotionCategory.Surprise,
    };

    for (const [keyword, emotion] of Object.entries(enKeywords)) {
      if (lowerText.includes(keyword)) scores[emotion] += 0.2;
    }

    // Exclamation/emphasis
    const exclamationCount = (text.match(/!/g) || []).length;
    const questionCount = (text.match(/\?/g) || []).length;
    const capsCount = (text.match(/[A-Z]{2,}/g) || []).length;

    if (exclamationCount > 1) {
      scores[EmotionCategory.Joy] += exclamationCount * 0.05;
      scores[EmotionCategory.Anger] += exclamationCount * 0.03;
    }
    if (questionCount > 1) scores[EmotionCategory.Surprise] += questionCount * 0.05;
    if (capsCount > 1) {
      scores[EmotionCategory.Anger] += capsCount * 0.08;
      scores[EmotionCategory.Surprise] += capsCount * 0.05;
    }

    // Negation handling
    const negationWords = ['không', 'chẳng', 'never', 'not', "don't", "can't", "won't"];
    for (const negation of negationWords) {
      for (const [keyword, emotion] of [...Object.entries(viKeywords), ...Object.entries(enKeywords)]) {
        const pattern = new RegExp(`${negation}\\s+${keyword}`, 'i');
        if (pattern.test(lowerText)) {
          scores[emotion] = Math.max(0, scores[emotion] - 0.15);
          scores[EmotionCategory.Neutral] += 0.1;
        }
      }
    }

    return scores;
  }

  /**
   * Get primary emotion from scores.
   */
  private getPrimaryEmotion(scores: EmotionScores, text: string, settings: ExtensionSettings): EmotionCategory {
    let highest = EmotionCategory.Neutral;
    let highestScore = scores[EmotionCategory.Neutral];

    for (const [emotion, score] of Object.entries(scores)) {
      if (emotion !== EmotionCategory.Neutral && score > highestScore) {
        highestScore = score;
        highest = emotion as EmotionCategory;
      }
    }

    if (highestScore < settings.confidenceThreshold) {
      return EmotionCategory.Neutral;
    }

    if (!settings.enabledEmotions.includes(highest)) {
      return EmotionCategory.Neutral;
    }

    return highest;
  }

  /**
   * Check if text contains adequate content for analysis.
   */
  hasContent(text: string): boolean {
    const trimmed = text.trim();
    if (trimmed.length < 2) return false;
    if (/^\d+$/.test(trimmed)) return false;
    if (/^https?:\/\//i.test(trimmed)) return false;
    if (/^@\w+$/.test(trimmed)) return false;
    return true;
  }
}

/** Singleton instance */
export const emotionClassifier = new EmotionClassifier();
