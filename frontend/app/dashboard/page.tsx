type SavedText = {
  id: string;
  user_id: string;
  source_text: string;
  source_url?: string | null;
  predicted_issue?: string | null;
  confidence: number;
  created_at: string;
};

type AggregatedAnalysis = {
  total_snippets: number;
  dominant_issue: string;
  average_confidence: number;
  risk_level: 'Low' | 'Medium' | 'High' | string;
  ai_summary: string;
};

type DashboardData = {
  saved_texts: SavedText[];
  aggregated_analysis: AggregatedAnalysis;
};

const emptyAnalysis: AggregatedAnalysis = {
  total_snippets: 0,
  dominant_issue: 'No data',
  average_confidence: 0,
  risk_level: 'Low',
  ai_summary:
    'Chưa có đủ dữ liệu đã lưu để tạo xu hướng cá nhân. Emotion Lens chỉ cung cấp thông tin tham khảo, không thay thế chẩn đoán hoặc tư vấn y khoa.',
};

async function getUserHistory(uid: string): Promise<DashboardData | null> {
  try {
    const res = await fetch(`http://localhost:8000/api/dashboard?uid=${encodeURIComponent(uid)}`, {
      cache: 'no-store',
    });

    if (!res.ok) {
      return null;
    }

    const payload = await res.json();

    if (Array.isArray(payload)) {
      return {
        saved_texts: payload,
        aggregated_analysis: emptyAnalysis,
      };
    }

    return payload as DashboardData;
  } catch {
    return null;
  }
}

function formatDate(value: string) {
  const date = new Date(value);

  if (Number.isNaN(date.getTime())) {
    return 'Không rõ thời gian';
  }

  return new Intl.DateTimeFormat('vi-VN', {
    day: '2-digit',
    month: '2-digit',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  }).format(date);
}

function formatPercent(value: number) {
  return `${Math.round((Number.isFinite(value) ? value : 0) * 100)}%`;
}

function normalizeIssue(issue?: string | null) {
  return issue?.trim() || 'Unknown';
}

function issueTone(issue?: string | null) {
  const normalized = normalizeIssue(issue).toLowerCase();

  if (normalized.includes('stress')) {
    return {
      border: 'border-l-orange-500',
      text: 'text-orange-300',
      badge: 'bg-orange-500/10 text-orange-200 ring-orange-400/30',
    };
  }

  if (normalized.includes('anxiety') || normalized.includes('anxious')) {
    return {
      border: 'border-l-yellow-400',
      text: 'text-yellow-200',
      badge: 'bg-yellow-500/10 text-yellow-100 ring-yellow-300/30',
    };
  }

  if (normalized.includes('depression')) {
    return {
      border: 'border-l-blue-500',
      text: 'text-blue-300',
      badge: 'bg-blue-500/10 text-blue-200 ring-blue-400/30',
    };
  }

  if (normalized.includes('suicidal')) {
    return {
      border: 'border-l-red-600',
      text: 'text-red-300',
      badge: 'bg-red-500/10 text-red-200 ring-red-400/30',
    };
  }

  if (normalized.includes('bipolar')) {
    return {
      border: 'border-l-violet-500',
      text: 'text-violet-300',
      badge: 'bg-violet-500/10 text-violet-200 ring-violet-400/30',
    };
  }

  if (normalized.includes('personality')) {
    return {
      border: 'border-l-pink-500',
      text: 'text-pink-300',
      badge: 'bg-pink-500/10 text-pink-200 ring-pink-400/30',
    };
  }

  if (normalized.includes('normal')) {
    return {
      border: 'border-l-slate-500',
      text: 'text-slate-300',
      badge: 'bg-slate-700/60 text-slate-200 ring-slate-500/40',
    };
  }

  return {
    border: 'border-l-slate-600',
    text: 'text-slate-300',
    badge: 'bg-slate-800 text-slate-200 ring-slate-600/50',
  };
}

function riskTone(riskLevel: string) {
  if (riskLevel === 'High') {
    return 'text-red-300 bg-red-500/10 ring-red-400/30';
  }

  if (riskLevel === 'Medium') {
    return 'text-amber-200 bg-amber-500/10 ring-amber-300/30';
  }

  return 'text-emerald-200 bg-emerald-500/10 ring-emerald-400/30';
}

function truncateMiddle(value: string, maxLength = 68) {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, 34)}...${value.slice(-22)}`;
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

function SnippetCard({ item }: { item: SavedText }) {
  const issue = normalizeIssue(item.predicted_issue);
  const tone = issueTone(issue);
  const sourceUrl = item.source_url?.trim();

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
            <p className="mt-1 text-xs text-slate-500">Không có source URL</p>
          )}
        </div>

        <div
          className={`inline-flex shrink-0 items-center justify-center rounded-md px-3 py-1 text-xs font-semibold ring-1 ${tone.badge}`}
        >
          {issue} · {formatPercent(item.confidence)}
        </div>
      </div>

      <p className="mt-4 whitespace-pre-wrap text-sm leading-6 text-slate-300">{item.source_text}</p>
    </article>
  );
}

export default async function DashboardPage({
  searchParams,
}: {
  searchParams: { [key: string]: string | string[] | undefined };
}) {
  const uidParam = searchParams.uid;
  const uid = Array.isArray(uidParam) ? uidParam[0] : uidParam;

  if (!uid) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
        <section className="mx-auto max-w-3xl rounded-lg border border-red-500/30 bg-red-500/10 p-6 text-center">
          <p className="text-lg font-semibold text-red-200">Thiếu User ID</p>
          <p className="mt-2 text-sm text-red-100/80">
            Vui lòng mở dashboard từ popup Emotion Lens để hệ thống truyền đúng mã người dùng.
          </p>
        </section>
      </main>
    );
  }

  const data = await getUserHistory(uid);

  if (!data) {
    return (
      <main className="min-h-screen bg-slate-950 px-6 py-16 text-slate-100">
        <section className="mx-auto max-w-3xl rounded-lg border border-slate-800 bg-slate-900 p-6 text-center">
          <p className="text-lg font-semibold text-slate-100">Không tải được dashboard</p>
          <p className="mt-2 text-sm text-slate-400">
            Backend chưa phản hồi hoặc UID không hợp lệ. Hãy kiểm tra FastAPI tại localhost:8000.
          </p>
        </section>
      </main>
    );
  }

  const { saved_texts: savedTexts, aggregated_analysis: analysis } = data;

  return (
    <main className="min-h-screen bg-slate-950 text-slate-100">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-8 px-5 py-8 sm:px-8 lg:px-10">
        <header className="flex flex-col gap-5 border-b border-slate-800 pb-6 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="flex flex-wrap items-center gap-3">
              <h1 className="text-3xl font-bold tracking-normal text-white md:text-4xl">
                Emotion Lens Analytics Dashboard
              </h1>
              <span className="rounded-md bg-violet-950 px-2.5 py-1 text-xs font-semibold text-violet-200 ring-1 ring-violet-500/30">
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
            accent={analysis.risk_level === 'High' ? 'text-red-300' : analysis.risk_level === 'Medium' ? 'text-amber-200' : 'text-emerald-200'}
          />
        </section>

        <section className="flex flex-col gap-4">
          <div className="flex flex-col gap-1 sm:flex-row sm:items-end sm:justify-between">
            <div>
              <h2 className="text-xl font-semibold text-white">Saved Snippets</h2>
              <p className="text-sm text-slate-500">Lịch sử văn bản đã lưu, mới nhất ở trên cùng.</p>
            </div>
            <p className="text-sm text-slate-500">{savedTexts.length} records</p>
          </div>

          {savedTexts.length > 0 ? (
            <div className="grid gap-4">
              {savedTexts.map((item) => (
                <SnippetCard key={item.id} item={item} />
              ))}
            </div>
          ) : (
            <div className="rounded-lg border border-slate-800 bg-slate-900 p-8 text-center">
              <p className="text-base font-semibold text-slate-200">Chưa có snippet nào được lưu</p>
              <p className="mt-2 text-sm text-slate-500">
                Khi bạn lưu văn bản từ Chrome Extension, dữ liệu sẽ xuất hiện tại đây.
              </p>
            </div>
          )}
        </section>

        <section className="rounded-lg border border-indigo-400/30 bg-indigo-500/10 p-6 shadow-2xl shadow-indigo-950/20">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div>
              <p className="text-sm font-semibold uppercase tracking-wide text-indigo-300">AI Recommendation</p>
              <h2 className="mt-2 text-xl font-semibold text-white">Safety Summary</h2>
            </div>
            <span className={`w-fit rounded-md px-3 py-1 text-xs font-semibold ring-1 ${riskTone(analysis.risk_level)}`}>
              {analysis.risk_level}
            </span>
          </div>
          <p className="mt-4 text-sm leading-6 text-indigo-50/90">{analysis.ai_summary}</p>
        </section>
      </div>
    </main>
  );
}
