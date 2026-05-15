// ====================================================
// AI-Powered Emotion Classifier (Local Inference)
// Uses Transformers.js with XLM-RoBERTa for multilingual
// emotion detection in Vietnamese and English
// ====================================================

import { EmotionCategory, EmotionResult, EmotionScores, ExtensionSettings } from '../types/emotion';

/**
 * Lightweight local emotion classifier using transformer models.
 * Designed for browser-based inference with ONNX runtime.
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
    '😡': EmotionCategory.Anger,
    '🤬': EmotionCategory.Anger,
    '😢': EmotionCategory.Sadness,
    '😭': EmotionCategory.Sadness,
    '😞': EmotionCategory.Sadness,
    '😔': EmotionCategory.Sadness,
    '😊': EmotionCategory.Joy,
    '😍': EmotionCategory.Joy,
    '🥰': EmotionCategory.Joy,
    '😂': EmotionCategory.Joy,
    '🤣': EmotionCategory.Joy,
    '🎉': EmotionCategory.Joy,
    '✨': EmotionCategory.Joy,
    '💀': EmotionCategory.Sarcastic,
    '🗿': EmotionCategory.Sarcastic,
    '😰': EmotionCategory.Anxiety,
    '😨': EmotionCategory.Fear,
    '😱': EmotionCategory.Fear,
    '😲': EmotionCategory.Surprise,
    '🤯': EmotionCategory.Surprise,
    '🙂': EmotionCategory.Sarcastic,
    '😏': EmotionCategory.Sarcastic,
    '😒': EmotionCategory.Sarcastic,
    '🤔': EmotionCategory.Surprise,
    '😐': EmotionCategory.Neutral,
    '👍': EmotionCategory.Joy,
    '❤️': EmotionCategory.Joy,
    '💔': EmotionCategory.Sadness,
    '☠️': EmotionCategory.Toxic,
    '🤮': EmotionCategory.Toxic,
    '👎': EmotionCategory.Anger,
    '😤': EmotionCategory.Anger,
    '😩': EmotionCategory.Sadness,
    '🥺': EmotionCategory.Sadness,
    '🤡': EmotionCategory.Sarcastic,
    '👀': EmotionCategory.Surprise,
    '🔥': EmotionCategory.Joy,
    '💯': EmotionCategory.Joy,
  };

  /** Punctuation patterns indicative of sarcasm */
  private sarcasmPatterns = [
    /~.*~/,
    /🤡/,
    /🙂$/,
    /sure,? .*!/i,
    /oh,? really/i,
    /wow,? .*!/i,
    /great,? .* not/i,
    /nice,? .* sarcasm/i,
    /obviously/i,
    /clearly/i,
    /totally not/i,
    /as if/i,
    /yeah,? right/i,
    /whatever you say/i,
    /hay quá ha/i,
    /giỏi quá ha/i,
    /tốt quá ha/i,
    /thông minh quá/i,
  ];

  /** Toxic pattern indicators */
  private toxicPatterns = [
    /\bstupid\b/i,
    /\bidiot\b/i,
    /\bkill yourself\b/i,
    /\bhate\b/i,
    /\btrash\b/i,
    /\bđồ ngu\b/i,
    /\bđi chết\b/i,
    /\bmày\b.*\bchết\b/i,
    /\bngu\b/i,
    /\bóc chó\b/i,
  ];

  constructor() {
    this.initializeSlangDictionary();
    this.initializeEmojiMap();
  }

  private initializeSlangDictionary(): void {
    // Merge Vietnamese and English slang
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

  /**
   * Initialize the transformer model pipeline.
   * Falls back to rule-based analysis if model loading fails.
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;
    if (this.initializationPromise) return this.initializationPromise;

    this.initializationPromise = this._initialize();
    return this.initializationPromise;
  }

  private async _initialize(): Promise<void> {
    try {
      // Try to load Transformers.js pipeline
      // This uses XLM-RoBERTa fine-tuned for emotion
      const { pipeline } = await import('@xenova/transformers');
      
      // Use a smaller, faster model for browser inference
      // XLM-RoBERTa-base works well for multilingual emotion
      this.model = await pipeline('text-classification', 'Xenova/xlm-roberta-base-emotion');
      this.tokenizer = null; // Handled by pipeline
      
      this.initialized = true;
      console.log('[EmotionLens] Transformer model loaded successfully');
    } catch (error) {
      console.warn('[EmotionLens] Could not load transformer model, using rule-based fallback:', error);
      // We still mark as initialized - will use rule-based only
      this.initialized = true;
    }
  }

  /**
   * Analyze text and return emotion classification result.
   * Uses multi-stage approach:
   * 1. Fast rule-based analysis (slang, emoji, pattern matching)
   * 2. Transformer model inference (if available)
   * 3. Ensemble scoring
   */
  async analyze(text: string, settings: ExtensionSettings): Promise<EmotionResult> {
    const startTime = performance.now();
    
    // Stage 1: Fast rule-based analysis
    const ruleBasedResult = this.ruleBasedAnalysis(text);
    
    // Stage 2: Transformer inference (if available)
    let transformerResult: number[] | null = null;
    if (this.model) {
      try {
        await this.initialize();
        transformerResult = await this.transformerInference(text);
      } catch (error) {
        console.warn('[EmotionLens] Transformer inference failed:', error);
      }
    }
    
    // Stage 3: Ensemble scoring
    const finalResult = this.ensembleScoring(
      ruleBasedResult,
      transformerResult,
      text,
      settings
    );
    
    finalResult.inferenceTimeMs = performance.now() - startTime;
    
    return finalResult;
  }

  /**
   * Fast rule-based emotion analysis.
   * Detects slang, emojis, patterns, and keywords.
   */
  private ruleBasedAnalysis(text: string): EmotionScores {
    const scores: EmotionScores = {
      [EmotionCategory.Joy]: 0,
      [EmotionCategory.Anger]: 0,
      [EmotionCategory.Sadness]: 0,
      [EmotionCategory.Anxiety]: 0,
      [EmotionCategory.Fear]: 0,
      [EmotionCategory.Surprise]: 0,
      [EmotionCategory.Neutral]: 0.2, // Default slight neutral
      [EmotionCategory.Toxic]: 0,
      [EmotionCategory.Sarcastic]: 0,
    };

    const lowerText = text.toLowerCase();
    const words = lowerText.split(/\s+/);
    const totalWords = words.length || 1;

    // --- Slang detection ---
    // Check multi-word phrases first
    for (const [phrase, emotion] of this.slangDictionary.entries()) {
      if (lowerText.includes(phrase)) {
        scores[emotion] += 0.4;
        scores[EmotionCategory.Neutral] = Math.max(0, scores[EmotionCategory.Neutral] - 0.15);
      }
    }

    // --- Emoji analysis ---
    // Count emojis and their emotional valence
    let emojiCount = 0;
    for (const char of text) {
      const emotion = this.emojiEmotionMap.get(char);
      if (emotion) {
        scores[emotion] += 0.3;
        emojiCount++;
      }
    }

    // --- Sarcasm detection ---
    for (const pattern of this.sarcasmPatterns) {
      if (pattern.test(text)) {
        scores[EmotionCategory.Sarcastic] += 0.45;
        scores[EmotionCategory.Neutral] -= 0.1;
      }
    }

    // --- Toxicity detection ---
    for (const pattern of this.toxicPatterns) {
      if (pattern.test(text)) {
        scores[EmotionCategory.Toxic] += 0.5;
        scores[EmotionCategory.Anger] += 0.2;
        scores[EmotionCategory.Neutral] -= 0.15;
      }
    }

    // --- Keyword analysis for Vietnamese ---
    const viKeywords: Record<string, EmotionCategory> = {
      'tuyệt': EmotionCategory.Joy,
      'vui': EmotionCategory.Joy,
      'hạnh': EmotionCategory.Joy,
      'phúc': EmotionCategory.Joy,
      'yêu': EmotionCategory.Joy,
      'thích': EmotionCategory.Joy,
      'cười': EmotionCategory.Joy,
      'ghét': EmotionCategory.Anger,
      'xấu': EmotionCategory.Sadness,
      'khóc': EmotionCategory.Sadness,
      'đau': EmotionCategory.Sadness,
      'buồn': EmotionCategory.Sadness,
      'chán': EmotionCategory.Sadness,
      'mệt': EmotionCategory.Sadness,
      'lo': EmotionCategory.Anxiety,
      'sợ': EmotionCategory.Fear,
      'ngại': EmotionCategory.Anxiety,
      'không biết': EmotionCategory.Surprise,
      'trời ơi': EmotionCategory.Surprise,
      'chết': EmotionCategory.Fear,
      'kinh': EmotionCategory.Surprise,
      'ghê': EmotionCategory.Surprise,
    };

    for (const [keyword, emotion] of Object.entries(viKeywords)) {
      if (lowerText.includes(keyword)) {
        scores[emotion] += 0.2;
      }
    }

    // --- English keyword analysis ---
    const enKeywords: Record<string, EmotionCategory> = {
      'happy': EmotionCategory.Joy,
      'love': EmotionCategory.Joy,
      'great': EmotionCategory.Joy,
      'amazing': EmotionCategory.Joy,
      'wonderful': EmotionCategory.Joy,
      'excited': EmotionCategory.Joy,
      'awesome': EmotionCategory.Joy,
      'fantastic': EmotionCategory.Joy,
      'beautiful': EmotionCategory.Joy,
      'angry': EmotionCategory.Anger,
      'mad': EmotionCategory.Anger,
      'furious': EmotionCategory.Anger,
      'hate': EmotionCategory.Anger,
      'terrible': EmotionCategory.Sadness,
      'awful': EmotionCategory.Sadness,
      'sad': EmotionCategory.Sadness,
      'depressed': EmotionCategory.Sadness,
      'cry': EmotionCategory.Sadness,
      'lonely': EmotionCategory.Sadness,
      'hurt': EmotionCategory.Sadness,
      'anxious': EmotionCategory.Anxiety,
      'worried': EmotionCategory.Anxiety,
      'nervous': EmotionCategory.Anxiety,
      'scared': EmotionCategory.Fear,
      'afraid': EmotionCategory.Fear,
      'terrified': EmotionCategory.Fear,
      'shocked': EmotionCategory.Surprise,
      'surprised': EmotionCategory.Surprise,
      'wow': EmotionCategory.Surprise,
      'omg': EmotionCategory.Surprise,
      'wtf': EmotionCategory.Surprise,
    };

    for (const [keyword, emotion] of Object.entries(enKeywords)) {
      if (lowerText.includes(keyword)) {
        scores[emotion] += 0.2;
      }
    }

    // --- Exclamation/emphasis detection ---
    const exclamationCount = (text.match(/!/g) || []).length;
    const questionCount = (text.match(/\?/g) || []).length;
    const capsCount = (text.match(/[A-Z]{2,}/g) || []).length;

    if (exclamationCount > 1) {
      scores[EmotionCategory.Joy] += exclamationCount * 0.05;
      scores[EmotionCategory.Anger] += exclamationCount * 0.03;
    }
    if (questionCount > 1) {
      scores[EmotionCategory.Surprise] += questionCount * 0.05;
    }
    if (capsCount > 1) {
      scores[EmotionCategory.Anger] += capsCount * 0.08;
      scores[EmotionCategory.Surprise] += capsCount * 0.05;
    }

    // --- Negation handling ---
    const negationWords = ['không', 'chẳng', 'never', 'not', "don't", "can't", "won't"];
    const emotionalWords = [...Object.keys(viKeywords), ...Object.keys(enKeywords)];
    
    for (const negation of negationWords) {
      for (const word of emotionalWords) {
        const pattern = new RegExp(`${negation}\\s+${word}`, 'i');
        if (pattern.test(lowerText)) {
          // Invert the emotion for negated terms
          const emotion = viKeywords[word] || enKeywords[word];
          scores[emotion] = Math.max(0, scores[emotion] - 0.15);
          scores[EmotionCategory.Neutral] += 0.1;
        }
      }
    }

    // --- Code-switching detection ---
    const viChars = (lowerText.match(/[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]/g) || []).length;
    const detectedLang = viChars > 0 ? 'vi' : 'en';
    
    // If mixed language detected, boost sarcasm probability
    if (detectedLang === 'vi' && lowerText.match(/[a-z]/g)) {
      const enWords = words.filter(w => /^[a-z]+$/.test(w)).length;
      if (enWords > 0 && enWords / totalWords > 0.1) {
        scores[EmotionCategory.Sarcastic] += 0.1;
      }
    }

    return scores;
  }

  /**
   * Run transformer model inference.
   * Returns logits/scores for each emotion category.
   */
  private async transformerInference(text: string): Promise<number[] | null> {
    if (!this.model) return null;
    
    try {
      const result = await this.model(text);
      // Transformers.js returns [{ label: string, score: number }]
      const scores: number[] = new Array(9).fill(0);
      
      if (Array.isArray(result)) {
        for (const item of result) {
          const labelIndex = this.mapLabelToIndex(item.label);
          if (labelIndex >= 0) {
            scores[labelIndex] = item.score;
          }
        }
      }
      
      return scores;
    } catch (error) {
      console.error('[EmotionLens] Transformer inference error:', error);
      return null;
    }
  }

  /**
   * Map transformer model label to emotion index.
   */
  private mapLabelToIndex(label: string): number {
    const labelMap: Record<string, number> = {
      'joy': 0,
      'anger': 1,
      'sadness': 2,
      'anxiety': 3,
      'fear': 4,
      'surprise': 5,
      'neutral': 6,
      'toxic': 7,
      'sarcastic': 8,
    };
    return labelMap[label.toLowerCase()] ?? -1;
  }

  /**
   * Ensemble scoring: combine rule-based and transformer results.
   */
  private ensembleScoring(
    ruleBased: EmotionScores,
    transformerResult: number[] | null,
    text: string,
    settings: ExtensionSettings
  ): EmotionResult {
    const weightedScores: EmotionScores = { ...ruleBased };
    
    // If transformer result available, blend with rule-based
    if (transformerResult) {
      const weight = 0.6; // Transformer weight
      const emotions = Object.values(EmotionCategory);
      for (let i = 0; i < emotions.length && i < transformerResult.length; i++) {
        const emotion = emotions[i];
        weightedScores[emotion] = 
          (ruleBased[emotion] * (1 - weight)) + 
          (transformerResult[i] * weight);
      }
    }

    // Apply sensitivity adjustment
    const sensitivityFactor = settings.sensitivity;
    for (const emotion of Object.values(EmotionCategory)) {
      if (emotion !== EmotionCategory.Neutral) {
        weightedScores[emotion] *= (0.5 + sensitivityFactor);
      }
    }

    // Normalize scores to 0-1
    const maxScore = Math.max(...Object.values(weightedScores), 0.01);
    for (const emotion of Object.values(EmotionCategory)) {
      weightedScores[emotion] /= maxScore;
      weightedScores[emotion] = Math.min(1, Math.max(0, weightedScores[emotion]));
    }

    // Ensure neutral is at least a small value
    weightedScores[EmotionCategory.Neutral] = Math.max(0.05, weightedScores[EmotionCategory.Neutral]);

    // Determine primary emotion
    let primaryEmotion = EmotionCategory.Neutral;
    let highestConfidence = 0;

    for (const [emotion, score] of Object.entries(weightedScores)) {
      if (score > highestConfidence && emotion !== EmotionCategory.Neutral) {
        highestConfidence = score;
        primaryEmotion = emotion as EmotionCategory;
      }
    }

    // Apply confidence threshold
    if (highestConfidence < settings.confidenceThreshold) {
      primaryEmotion = EmotionCategory.Neutral;
      highestConfidence = weightedScores[EmotionCategory.Neutral];
    }

    // Check if user has this emotion enabled
    if (!settings.enabledEmotions.includes(primaryEmotion)) {
      primaryEmotion = EmotionCategory.Neutral;
    }

    return {
      primaryEmotion,
      scores: weightedScores,
      confidence: highestConfidence,
      toxicityScore: weightedScores[EmotionCategory.Toxic],
      sarcasmScore: weightedScores[EmotionCategory.Sarcastic],
      language: this.detectLanguage(text),
      source: 'local',
      inferenceTimeMs: 0, // Will be set by caller
    };
  }

  /**
   * Detect whether text is Vietnamese, English, or mixed.
   */
  private detectLanguage(text: string): 'vi' | 'en' | 'mixed' {
    const viChars = text.match(/[àáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵđ]/g);
    const viCount = viChars?.length || 0;
    const totalChars = text.replace(/\s/g, '').length;
    
    if (totalChars === 0) return 'en';
    
    const viRatio = viCount / totalChars;
    
    if (viRatio > 0.15) {
      // Check for significant English presence
      const enWords = text.match(/[a-z]+/gi)?.length || 0;
      const viSize = text.length;
      if (enWords > 0 && enWords / (viSize / 5) > 0.3) {
        return 'mixed';
      }
      return 'vi';
    }
    
    return 'en';
  }

  /**
   * Check if text contains adequate content for analysis.
   */
  hasContent(text: string): boolean {
    const trimmed = text.trim();
    if (trimmed.length < 2) return false;
    // Filter out pure numbers, URLs, etc.
    if (/^\d+$/.test(trimmed)) return false;
    if (/^https?:\/\//i.test(trimmed)) return false;
    if (/^@\w+$/.test(trimmed)) return false;
    return true;
  }
}

/** Singleton instance */
export const emotionClassifier = new EmotionClassifier();