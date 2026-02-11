from __future__ import annotations
from typing import List
from datetime import datetime
from shared.models import ThreadTriage

P_RANK = {"P0":0, "P1":1, "P2":2, "P3":3}

def render_summary_md(triages: List[ThreadTriage]) -> str:
    def best_rank(t: ThreadTriage) -> int:
        if not t.actions:
            return 99
        return min(P_RANK[a.priority] for a in t.actions)

    actionable = [t for t in triages if t.email_type=="ACTION_REQUIRED" and t.action_required]
    info = [t for t in triages if t.email_type=="INFORMATIONAL_ARCHIVE"]
    irr = [t for t in triages if t.email_type=="IRRELEVANT"]

    actionable.sort(key=best_rank)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Daily Workload Summary\n\n_Generated: {now}_\n"]

    total = len(triages)
    lines.append("## Summary\n")
    lines.append(f"- Total threads: **{total}**")
    lines.append(f"- Action required: **{len(actionable)}**")
    lines.append(f"- Informational/archive: **{len(info)}**")
    lines.append(f"- Irrelevant/noise: **{len(irr)}**\n")

    lines.append("## Action Required (prioritised)\n")
    if not actionable:
        lines.append("- None\n")
    else:
        for t in actionable:
            ref = t.thread_ref or "(no-ref)"
            topic = t.topic or "Unspecified"
            cp = (t.entities or {}).get("counterparty") if isinstance(t.entities, dict) else None
            cp_txt = f" | Counterparty: **{cp}**" if cp else ""
            handler_txt = f" | Handler: **{t.handler_name}**" if t.handler_name else ""
            cust_txt = f" | Customer: **{t.customer_name}**" if t.customer_name else ""
            lines.append(f"### {ref} — {topic}{cp_txt}{handler_txt}{cust_txt}\n")
            if t.urgency_reason:
                lines.append(f"- **Urgency:** {t.urgency_reason}")
            for a in t.actions:
                due = f" (Due: {a.due})" if a.due else ""
                blk = " [BLOCKING]" if a.blocking else ""
                owner = f" (Owner: {a.owner})" if a.owner else ""
                lines.append(f"- **{a.priority}**{blk}: {a.action_item}{due}{owner}")
            if t.missing_info:
                lines.append(f"- **Missing info:** {', '.join(t.missing_info)}")
            if t.rationale:
                lines.append(f"- **Why:** {t.rationale}")
            lines.append("")

    lines.append("## Informational (archive)\n")
    lines.extend([f"- {(t.thread_ref or '(no-ref)')} — {(t.topic or 'FYI')}" for t in info[:30]])
    if len(info) > 30:
        lines.append(f"\n_...and {len(info)-30} more._\n")

    lines.append("\n## Irrelevant / Noise\n")
    lines.extend([f"- {(t.thread_ref or '(no-ref)')} — {(t.topic or 'Noise')}" for t in irr[:20]])
    if len(irr) > 20:
        lines.append(f"\n_...and {len(irr)-20} more._\n")

    return "\n".join(lines)

def render_user_day_summary_md(triages: List[ThreadTriage], user_email: str, day: str) -> str:
    def best_rank(t: ThreadTriage) -> int:
        if not t.actions:
            return 99
        return min(P_RANK[a.priority] for a in t.actions)

    actionable = [t for t in triages if t.email_type=="ACTION_REQUIRED" and t.action_required]
    info = [t for t in triages if t.email_type=="INFORMATIONAL_ARCHIVE"]
    irr = [t for t in triages if t.email_type=="IRRELEVANT"]

    actionable.sort(key=best_rank)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [f"# Daily Summary — {user_email} — {day}\n\n_Generated: {now}_\n"]

    total = len(triages)
    lines.append("## Summary\n")
    lines.append(f"- Total threads: **{total}**")
    lines.append(f"- Action required: **{len(actionable)}**")
    lines.append(f"- Informational/archive: **{len(info)}**")
    lines.append(f"- Irrelevant/noise: **{len(irr)}**\n")

    lines.append("## Action Required (prioritised)\n")
    if not actionable:
        lines.append("- None\n")
    else:
        for t in actionable:
            ref = t.thread_ref or "(no-ref)"
            topic = t.topic or "Unspecified"
            cp = (t.entities or {}).get("counterparty") if isinstance(t.entities, dict) else None
            cp_txt = f" | Counterparty: **{cp}**" if cp else ""
            handler_txt = f" | Handler: **{t.handler_name}**" if t.handler_name else ""
            cust_txt = f" | Customer: **{t.customer_name}**" if t.customer_name else ""
            lines.append(f"### {ref} — {topic}{cp_txt}{handler_txt}{cust_txt}\n")
            if t.urgency_reason:
                lines.append(f"- **Urgency:** {t.urgency_reason}")
            for a in t.actions:
                due = f" (Due: {a.due})" if a.due else ""
                blk = " [BLOCKING]" if a.blocking else ""
                owner = f" (Owner: {a.owner})" if a.owner else ""
                lines.append(f"- **{a.priority}**{blk}: {a.action_item}{due}{owner}")
            if t.missing_info:
                lines.append(f"- **Missing info:** {', '.join(t.missing_info)}")
            if t.rationale:
                lines.append(f"- **Why:** {t.rationale}")
            lines.append("")

    lines.append("## Informational (archive)\n")
    lines.extend([f"- {(t.thread_ref or '(no-ref)')} — {(t.topic or 'FYI')}" for t in info[:30]])
    if len(info) > 30:
        lines.append(f"\n_...and {len(info)-30} more._\n")

    lines.append("\n## Irrelevant / Noise\n")
    lines.extend([f"- {(t.thread_ref or '(no-ref)')} — {(t.topic or 'Noise')}" for t in irr[:20]])
    if len(irr) > 20:
        lines.append(f"\n_...and {len(irr)-20} more._\n")

    return "\n".join(lines)
