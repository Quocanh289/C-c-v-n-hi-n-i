'use client';

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useSearchParams } from 'next/navigation';

type EmotionNuance = {
  key: string;
  label: string;
  score: number;
};

type DistributionItem = {
  key: string;
  label: string;
  ratio: number;
};

type PotentialCondition = {
  label: string;
  severity: 'low' | 'medium' | 'high' | 'info' | string;
};

type SavedText = {
  id: string;
  user_id: string;
  source_text: string;
  source_url?: string | null;
  predicted_issue?: string | null;
  confidence: number;
  created_at: string;
  emotion_nuances?: EmotionNuance[];
};

type AggregatedAnalysis = {
  total_snippets: number;
  dominant_issue: string;
  average_confidence: number;
  risk_level: 'Low' | 'Medium' | 'High' | string;
  ai_summary: string;
  emotion_distribution: DistributionItem[];
  issue_distribution: DistributionItem[];
  potential_conditions: PotentialCondition[];
  recommendations: string[];
  disclaimer: string;
};

type DashboardData = {
  saved_texts: SavedText[];
  aggregated_analysis: AggregatedAnalysis;
};

type ToastState = {
  visible: boolean;
  message: string;
  tone: 'success' | 'error';
};

const emptyAnalysis: AggregatedAnalysis = {
  total_snippets: 0,
  dominant_issue: 'No data',
  average_confidence: 0,
  risk_level: 'Low',
  ai_summary:
    'No saved snippets yet, so a personalized trend cannot be computed. This dashboard is informational only and cannot replace professional diagnosis or treatment.',
  emotion_distribution: [{ key: 'neutral', label: 'Neutral', ratio: 1 }],
  issue_distribution: [{ key: 'normal', label: 'Normal', ratio: 1 }],
  potential_conditions: [{ label: 'No data yet', severity: 'info' }],
  recommendations: [
    'Save a few snippets to unlock personalized recommendations.',
    'Keep logging short reflections to capture trends over time.',
  ],
  disclaimer:
    'Safety notice: this is not a medical diagnosis. If you feel at risk of harming yourself or someone else, contact local emergency services immediately (911 in the United States) or your local crisis hotline.',
};

const EMOTION_VISUALS: Record<string, { label: string; icon: string; color: string; bgColor: string }> = {
  admiration: { label: 'Admiration', icon: '\u{1F44F}', color: '#f59e0b', bgColor: 'rgba(245, 158, 11, 0.14)' },
  amusement: { label: 'Amusement', icon: '\u{1F602}', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.14)' },
  anger: { label: 'Anger', icon: '\u{1F621}', color: '#ef4444', bgColor: 'rgba(239, 68, 68, 0.14)' },
  annoyance: { label: 'Annoyance', icon: '\u{1F624}', color: '#f97316', bgColor: 'rgba(249, 115, 22, 0.14)' },
  approval: { label: 'Approval', icon: '\u{1F44D}', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.14)' },
  caring: { label: 'Caring', icon: '\u{1F49A}', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.14)' },
  confusion: { label: 'Confusion', icon: '\u{1F615}', color: '#a855f7', bgColor: 'rgba(168, 85, 247, 0.14)' },
  curiosity: { label: 'Curiosity', icon: '\u{1F914}', color: '#a855f7', bgColor: 'rgba(168, 85, 247, 0.14)' },
  desire: { label: 'Desire', icon: '\u{1F60D}', color: '#ec4899', bgColor: 'rgba(236, 72, 153, 0.14)' },
  disappointment: { label: 'Disappointment', icon: '\u{1F61E}', color: '#3b82f6', bgColor: 'rgba(59, 130, 246, 0.14)' },
  disapproval: { label: 'Disapproval', icon: '\u{1F44E}', color: '#f97316', bgColor: 'rgba(249, 115, 22, 0.14)' },
  disgust: { label: 'Disgust', icon: '\u{1F922}', color: '#84cc16', bgColor: 'rgba(132, 204, 22, 0.14)' },
  embarrassment: { label: 'Embarrassment', icon: '\u{1F633}', color: '#f472b6', bgColor: 'rgba(244, 114, 182, 0.14)' },
  excitement: { label: 'Excitement', icon: '\u{1F929}', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.14)' },
  fear: { label: 'Fear', icon: '\u{1F628}', color: '#7c3aed', bgColor: 'rgba(124, 58, 237, 0.14)' },
  gratitude: { label: 'Gratitude', icon: '\u{1F64F}', color: '#f59e0b', bgColor: 'rgba(245, 158, 11, 0.14)' },
  grief: { label: 'Grief', icon: '\u{1F62D}', color: '#3b82f6', bgColor: 'rgba(59, 130, 246, 0.14)' },
  joy: { label: 'Joy', icon: '\u{1F60A}', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.14)' },
  love: { label: 'Love', icon: '\u{2764}\u{FE0F}', color: '#ef4444', bgColor: 'rgba(239, 68, 68, 0.14)' },
  nervousness: { label: 'Nervousness', icon: '\u{1F62C}', color: '#f97316', bgColor: 'rgba(249, 115, 22, 0.14)' },
  optimism: { label: 'Optimism', icon: '\u{1F31F}', color: '#f59e0b', bgColor: 'rgba(245, 158, 11, 0.14)' },
  pride: { label: 'Pride', icon: '\u{1F981}', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.14)' },
  realization: { label: 'Realization', icon: '\u{1F4A1}', color: '#a855f7', bgColor: 'rgba(168, 85, 247, 0.14)' },
  relief: { label: 'Relief', icon: '\u{1F60C}', color: '#22c55e', bgColor: 'rgba(34, 197, 94, 0.14)' },
  remorse: { label: 'Remorse', icon: '\u{1F614}', color: '#3b82f6', bgColor: 'rgba(59, 130, 246, 0.14)' },
  sadness: { label: 'Sadness', icon: '\u{1F61E}', color: '#3b82f6', bgColor: 'rgba(59, 130, 246, 0.14)' },
  surprise: { label: 'Surprise', icon: '\u{1F632}', color: '#a855f7', bgColor: 'rgba(168, 85, 247, 0.14)' },
  neutral: { label: 'Neutral', icon: '\u{1F610}', color: '#6b7280', bgColor: 'rgba(107, 114, 128, 0.18)' },
  anxiety: { label: 'Anxiety', icon: '\u{1F630}', color: '#f97316', bgColor: 'rgba(249, 115, 22, 0.14)' },
};

const ISSUE_VISUALS: Record<string, { label: string; color: string }> = {
  normal: { label: 'Normal', color: '#6b7280' },
  stress: { label: 'Stress', color: '#f97316' },
  anxiety: { label: 'Anxiety', color: '#f59e0b' },
  depression: { label: 'Depression', color: '#3b82f6' },
  bipolar: { label: 'Bipolar', color: '#8b5cf6' },
  personality_disorder: { label: 'Personality Disorder', color: '#ec4899' },
  suicidal: { label: 'Suicidal', color: '#ef4444' },
  unknown: { label: 'Unknown', color: '#94a3b8' },
};

const CONDITION_TONES: Record<string, string> = {
  low: 'bg-amber-500/15 text-amber-100 ring-amber-300/40',
  medium: 'bg-orange-500/15 text-orange-100 ring-orange-300/40',
  high: 'bg-rose-500/20 text-rose-100 ring-rose-300/45',
  info: 'bg-slate-500/20 text-slate-100 ring-slate-300/35',
};

function normalizeIssue(issue?: string | null) {
  return issue?.trim() || 'Unknown';
}

function issueTone(issue?: string | null) {
  const normalized = normalizeIssue(issue).toLowerCase();

  if (normalized.includes('stress')) {
    return {
      border: 'border-l-orange-500',
      text: 'text-orange-300',
    };
  }
  if (normalized.includes('anxiety') || normalized.includes('anxious')) {
    return {
      border: 'border-l-yellow-400',
      text: 'text-yellow-200',
    };
  }
  if (normalized.includes('depression')) {
    return {
      border: 'border-l-blue-500',
      text: 'text-blue-300',
    };
  }
  if (normalized.includes('suicidal')) {
    return {
      border: 'border-l-red-500',
      text: 'text-red-300',
    };
  }
  if (normalized.includes('bipolar')) {
    return {
      border: 'border-l-violet-500',
      text: 'text-violet-300',
    };
  }
  if (normalized.includes('personality')) {
    return {
      border: 'border-l-pink-500',
      text: 'text-pink-300',
    };
  }
  if (normalized.includes('normal')) {
    return {
      border: 'border-l-slate-500',
      text: 'text-slate-300',
    };
  }

  return {
    border: 'border-l-slate-600',
    text: 'text-slate-300',
  };
}

function riskTone(riskLevel: string) {
  if (riskLevel === 'High') {
    return 'text-red-200 bg-red-500/15 ring-red-300/45';
  }
  if (riskLevel === 'Medium') {
    return 'text-amber-100 bg-amber-500/15 ring-amber-300/45';
  }
  return 'text-emerald-100 bg-emerald-500/15 ring-emerald-300/45';
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return 'Unknown time';
  }

  return new Intl.DateTimeFormat('en-US', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);
}

function formatPercent(value: number, digits = 1) {
  const safeValue = Number.isFinite(value) ? value : 0;
  return `${(safeValue * 100).toFixed(digits)}%`;
}

function truncateMiddle(value: string, maxLength = 72) {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, 36)}...${value.slice(-22)}`;
}

function getNuances(item: SavedText): EmotionNuance[] {
  const source = Array.isArray(item.emotion_nuances) ? item.emotion_nuances : [];
  if (source.length === 0) {
    return [{ key: 'neutral', label: 'Neutral', score: 0.35 }];
  }

  return [...source]
    .sort((a, b) => (b.score ?? 0) - (a.score ?? 0))
    .slice(0, 4)
    .map((entry) => ({
      key: String(entry.key || '').toLowerCase(),
      label: entry.label || EMOTION_VISUALS[entry.key]?.label || 'Neutral',
      score: Number.isFinite(entry.score) ? entry.score : 0,
    }));
}

function normalizeIssueKey(value: string) {
  return value.trim().toLowerCase().replace(/\s+/g, '_');
}

function resolveIssueVisual(value: string) {
  const key = normalizeIssueKey(value);
  return ISSUE_VISUALS[key] || ISSUE_VISUALS.unknown;
}

async function getUserHistory(uid: string): Promise<DashboardData | null> {
  const baseUrl = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

  try {
    const res = await fetch(`${baseUrl}/api/dashboard?uid=${encodeURIComponent(uid)}`, {
      cache: 'no-store',
    });

    if (!res.ok) {
      return null;
    }

    const payload = await res.json();
    if (Array.isArray(payload)) {
      return { saved_texts: payload, aggregated_analysis: emptyAnalysis };
    }

    return payload as DashboardData;
  } catch {
    return null;
  }
}

function StatCard({
  label,
  value,
  detail,
  accent = 'text-indigo-300',
}: {
  label: string;
  value: string;
  detail: string;
  accent?: string;
}) {
  return (
    <div className="rounded-lg border border-slate-800 bg-slate-900 p-4 shadow-xl shadow-slate-950/30">
      <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500">{label}</p>
      <p className={`mt-3 truncate text-2xl font-bold ${accent}`}>{value}</p>
      <p className="mt-1 truncate text-xs text-slate-400">{detail}</p>
    </div>
  );
}

function NuanceBadge({ nuance }: { nuance: EmotionNuance }) {
  const visual = EMOTION_VISUALS[nuance.key] || EMOTION_VISUALS.neutral;

  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-semibold"
      style={{
        color: visual.color,
        backgroundColor: visual.bgColor,
        borderColor: `${visual.color}66`,
      }}
      title={`${visual.label} - ${formatPercent(nuance.score)}`}
    >
      <span>{visual.icon}</span>
      <span>{visual.label}</span>
    </span>
  );
}

function ProgressRow({ label, value, color }: { label: string; value: number; color: string }) {
  const clamped = Math.min(Math.max(value, 0), 1);

  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between text-xs text-slate-300">
        <span className="truncate">{label}</span>
        <span className="text-slate-400">{formatPercent(clamped)}</span>
      </div>
      <div className="w-full rounded-full bg-slate-800 h-2.5">
        <div
          className="h-2.5 rounded-full"
          style={{ width: `${clamped * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function ConditionBadgeItem({ condition }: { condition: PotentialCondition }) {
  const tone = CONDITION_TONES[condition.severity] || CONDITION_TONES.info;

  return (
    <span className={`rounded-full px-2.5 py-1 text-[11px] font-semibold ring-1 ${tone}`}>
      {condition.label}
    </span>
  );
}

function RecommendationItem({ text }: { text: string }) {
  return (
    <li className="flex items-start gap-2 text-sm text-slate-200">
      <span className="mt-0.5 inline-flex h-5 w-5 items-center justify-center rounded-full bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-300/40">
        <svg viewBox="0 0 20 20" className="h-3.5 w-3.5" fill="currentColor" aria-hidden="true">
          <path
            fillRule="evenodd"
            d="M16.704 5.293a1 1 0 0 1 0 1.414l-7.415 7.414a1 1 0 0 1-1.414 0L3.296 9.543a1 1 0 1 1 1.414-1.414l3.165 3.165 6.708-6.708a1 1 0 0 1 1.414 0Z"
            clipRule="evenodd"
          />
        </svg>
      </span>
      <span className="leading-6">{text}</span>
    </li>
  );
}

function SnippetCard({ item }: { item: SavedText }) {
  const issue = normalizeIssue(item.predicted_issue);
  const tone = issueTone(issue);
  const sourceUrl = item.source_url?.trim();
  const nuances = getNuances(item);

  return (
    <article
      className={`rounded-lg border border-slate-800 bg-slate-900 p-5 shadow-xl shadow-slate-950/25 ${tone.border} border-l-4`}
    >
      <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <p className="text-sm font-medium text-slate-200">{formatDate(item.created_at)}</p>
          {sourceUrl ? (
            <a
              href={sourceUrl}
              target="_blank"
              rel="noreferrer"
              title={sourceUrl}
              className="mt-1 block max-w-full truncate text-xs text-indigo-300 transition hover:text-indigo-200"
            >
              {truncateMiddle(sourceUrl)}
            </a>
          ) : (
            <p className="mt-1 text-xs text-slate-500">No source URL</p>
          )}
        </div>

        <div className={`text-xs font-semibold ${tone.text}`}>{issue}</div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {nuances.map((nuance) => (
          <NuanceBadge key={`${item.id}-${nuance.key}-${nuance.label}`} nuance={nuance} />
        ))}
      </div>

      <p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-slate-300">{item.source_text}</p>
    </article>
  );
}

export default function DashboardPage() {
  const searchParams = useSearchParams();
  const uid = useMemo(() => searchParams.get('uid') || '', [searchParams]);

  const [data, setData] = useState<DashboardData | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [toast, setToast] = useState<ToastState>({ visible: false, message: '', tone: 'success' });
  const toastTimerRef = useRef<NodeJS.Timeout | null>(null);

  const showToast = useCallback((message: string, tone: 'success' | 'error') => {
    if (toastTimerRef.current) {
      clearTimeout(toastTimerRef.current);
    }

    setToast({ visible: true, message, tone });
    toastTimerRef.current = setTimeout(() => {
      setToast((prev) => ({ ...prev, visible: false }));
      toastTimerRef.current = null;
    }, 3000);
  }, []);

  const refreshHistory = useCallback(
    async (showFeedback: boolean) => {
      if (!uid) {
        return;
      }

      setIsLoading(true);
      const nextData = await getUserHistory(uid);
      setData(nextData);
      setIsLoading(false);
      setHasLoaded(true);

      if (showFeedback) {
        if (nextData) {
          showToast('History refreshed! Keep saving snippets to monitor your well-being.', 'success');
        } else {
          showToast('Refresh failed. Please check the backend connection and try again.', 'error');
        }
      }
    },
    [showToast, uid],
  );

  useEffect(() => {
    refreshHistory(false);

    return () => {
      if (toastTimerRef.current) {
        clearTimeout(toastTimerRef.current);
      }
    };
  }, [refreshHistory]);

  if (!uid) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
        <section className="mx-auto max-w-3xl rounded-lg border border-red-500/30 bg-red-500/10 p-6 text-center">
          <p className="text-lg font-semibold text-red-200">Missing User ID</p>
          <p className="mt-2 text-sm text-red-100/80">
            Please open this dashboard from the Emotion Lens popup so the correct user ID is attached.
          </p>
        </section>
      </main>
    );
  }

  if (!hasLoaded || isLoading) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
        <section className="mx-auto max-w-3xl rounded-lg border border-slate-800 bg-slate-900 p-6 text-center">
          <p className="text-lg font-semibold text-slate-100">Loading dashboard...</p>
          <p className="mt-2 text-sm text-slate-400">Fetching the latest emotional history from the API.</p>
        </section>
      </main>
    );
  }

  if (!data) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
        <section className="mx-auto max-w-3xl rounded-lg border border-slate-800 bg-slate-900 p-6 text-center">
          <p className="text-lg font-semibold text-slate-100">Dashboard unavailable</p>
          <p className="mt-2 text-sm text-slate-400">
            The backend is not responding or the UID is invalid. Please verify FastAPI at `localhost:8000`.
          </p>
        </section>
      </main>
    );
  }

  const { saved_texts: savedTexts, aggregated_analysis: analysis } = data;
  const emotionDistribution = analysis.emotion_distribution?.length
    ? analysis.emotion_distribution
    : emptyAnalysis.emotion_distribution;
  const issueDistribution = analysis.issue_distribution?.length
    ? analysis.issue_distribution
    : emptyAnalysis.issue_distribution;
  const potentialConditions = analysis.potential_conditions?.length
    ? analysis.potential_conditions
    : emptyAnalysis.potential_conditions;
  const recommendations = analysis.recommendations?.length
    ? analysis.recommendations
    : emptyAnalysis.recommendations;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      {toast.visible ? (
        <div
          className={`fixed right-5 top-5 z-50 rounded-lg border px-4 py-3 text-sm font-medium shadow-2xl backdrop-blur ${
            toast.tone === 'success'
              ? 'border-emerald-400/40 bg-emerald-500/15 text-emerald-100'
              : 'border-red-400/40 bg-red-500/15 text-red-100'
          }`}
        >
          {toast.message}
        </div>
      ) : null}

      <div className="mx-auto flex w-full max-w-7xl flex-col gap-7 px-5 py-8 sm:px-8 lg:px-10">
        <header className="flex flex-col gap-5 border-b border-slate-800 pb-6 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-3xl font-bold text-white md:text-4xl">Emotion Lens Analytics Dashboard</h1>
              <span className="rounded-md bg-violet-950 px-2.5 py-1 text-xs font-semibold text-violet-200 ring-1 ring-violet-500/35">
                v2.1
              </span>
            </div>
            <p className="mt-3 max-w-2xl break-all text-sm text-slate-400">User ID: {uid}</p>
          </div>

          <div className={`rounded-lg px-4 py-3 text-sm font-semibold ring-1 ${riskTone(analysis.risk_level)}`}>
            Overall Risk: {analysis.risk_level}
          </div>
        </header>

        <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatCard
            label="Total Snippets Saved"
            value={String(analysis.total_snippets)}
            detail="Saved from Chrome Extension"
          />
          <StatCard
            label="Primary Issue"
            value={analysis.dominant_issue}
            detail="Most frequent prediction"
            accent={issueTone(analysis.dominant_issue).text}
          />
          <StatCard
            label="Average Confidence"
            value={formatPercent(analysis.average_confidence)}
            detail="Mean model confidence"
          />
          <StatCard
            label="Overall Risk"
            value={analysis.risk_level}
            detail="Calculated from history"
            accent={
              analysis.risk_level === 'High'
                ? 'text-red-300'
                : analysis.risk_level === 'Medium'
                  ? 'text-amber-200'
                  : 'text-emerald-200'
            }
          />
        </section>

        <section className="grid gap-6 lg:grid-cols-12 lg:items-start">
          <div className="lg:col-span-8">
            <div className="rounded-lg border border-slate-800 bg-slate-900/70 p-5 shadow-xl shadow-slate-950/30">
              <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
                <div>
                  <h2 className="text-xl font-semibold text-white">Saved Snippets</h2>
                  <p className="text-sm text-slate-500">Your saved text history, newest first.</p>
                </div>
                <p className="text-sm text-slate-500">{savedTexts.length} records</p>
              </div>

              {savedTexts.length > 0 ? (
                <div className="mt-5 grid gap-4 lg:max-h-[calc(100vh-18rem)] lg:overflow-y-auto lg:pr-2">
                  {savedTexts.map((item) => (
                    <SnippetCard key={item.id} item={item} />
                  ))}
                </div>
              ) : (
                <div className="mt-5 rounded-lg border border-slate-800 bg-slate-900 p-8 text-center">
                  <p className="text-base font-semibold text-slate-200">No snippets saved yet</p>
                  <p className="mt-2 text-sm text-slate-500">
                    Save text from the Chrome Extension and it will appear here.
                  </p>
                </div>
              )}
            </div>
          </div>

          <aside className="lg:col-span-4 lg:sticky lg:top-6">
            <section className="relative overflow-hidden rounded-xl border-2 border-indigo-300/45 bg-gradient-to-br from-indigo-500/20 via-violet-500/15 to-slate-900 p-6 shadow-[0_0_50px_rgba(99,102,241,0.35)]">
              <div className="pointer-events-none absolute -right-12 -top-12 h-40 w-40 rounded-full bg-indigo-400/25 blur-3xl" />
              <div className="pointer-events-none absolute -left-10 bottom-0 h-32 w-32 rounded-full bg-violet-400/20 blur-3xl" />

              <div className="relative z-10">
                <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
                  <div>
                    <p className="text-sm font-semibold uppercase tracking-wide text-indigo-200">AI Recommendation</p>
                    <h2 className="mt-2 text-2xl font-semibold text-white">Safety Summary</h2>
                  </div>
                  <span className={`w-fit rounded-md px-3 py-1 text-xs font-semibold ring-1 ${riskTone(analysis.risk_level)}`}>
                    {analysis.risk_level}
                  </span>
                </div>

                <div className="mt-6 grid gap-4">
                  <div className="rounded-lg border border-indigo-400/25 bg-slate-900/70 p-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-semibold text-indigo-100">Emotion & Issue Distribution</h3>
                      <span className="text-[11px] text-slate-400">{analysis.total_snippets} samples</span>
                    </div>

                    <div className="mt-4 grid gap-4">
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Emotions</p>
                        <div className="mt-3 space-y-3">
                          {emotionDistribution.map((item) => {
                            const visual = EMOTION_VISUALS[item.key] || EMOTION_VISUALS.neutral;
                            return (
                              <ProgressRow
                                key={`emotion-${item.key}`}
                                label={item.label || visual.label}
                                value={item.ratio}
                                color={visual.color}
                              />
                            );
                          })}
                        </div>
                      </div>

                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-400">Issues</p>
                        <div className="mt-3 space-y-3">
                          {issueDistribution.map((item) => {
                            const visual = resolveIssueVisual(item.key || item.label);
                            return (
                              <ProgressRow
                                key={`issue-${item.key || item.label}`}
                                label={item.label || visual.label}
                                value={item.ratio}
                                color={visual.color}
                              />
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="rounded-lg border border-amber-400/25 bg-slate-900/70 p-4">
                    <h3 className="text-sm font-semibold text-amber-100">Potential Conditions</h3>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {potentialConditions.map((condition, index) => (
                        <ConditionBadgeItem key={`${condition.label}-${index}`} condition={condition} />
                      ))}
                    </div>
                  </div>

                  <div className="rounded-lg border border-emerald-400/20 bg-slate-900/70 p-4">
                    <h3 className="text-sm font-semibold text-emerald-100">Actionable Recommendations</h3>
                    <ul className="mt-3 space-y-2">
                      {recommendations.map((step, index) => (
                        <RecommendationItem key={`${step}-${index}`} text={step} />
                      ))}
                    </ul>
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => refreshHistory(true)}
                  disabled={isLoading}
                  className="mt-6 inline-flex w-full items-center justify-center rounded-md bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white transition hover:bg-indigo-500 disabled:cursor-not-allowed disabled:opacity-60"
                >
                  {isLoading ? 'Refreshing...' : 'Keep Monitoring'}
                </button>

                <p className="mt-4 text-[11px] leading-5 text-indigo-100/60">
                  {analysis.disclaimer || emptyAnalysis.disclaimer}
                </p>
              </div>
            </section>
          </aside>
        </section>
      </div>
    </main>
  );
}
