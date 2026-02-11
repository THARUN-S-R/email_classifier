from __future__ import annotations
import weaviate
from weaviate.classes.config import Property, DataType, Configure
from email_classifier.weaviate_service.weaviate_client import get_client
from email_classifier.shared.config import THREAD_CLASS, DETAIL_CLASS, DAILY_SUMMARY_CLASS

def ensure_schema():
    client = get_client()
    client.connect()
    try:
        existing_raw = client.collections.list_all()
        if isinstance(existing_raw, dict):
            existing = set(existing_raw.keys())
        elif isinstance(existing_raw, list):
            if existing_raw and isinstance(existing_raw[0], str):
                existing = set(existing_raw)
            else:
                existing = {c.name for c in existing_raw}
        else:
            existing = set()

        thread_props = [
            Property(name="thread_id", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="thread_key", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="thread_ref", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="email_type", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="topic", data_type=DataType.TEXT),
            Property(name="topic_lc", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="category", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="priority_best", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="action_required", data_type=DataType.BOOL),
            Property(name="actions_json", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="actions_text", data_type=DataType.TEXT),
            Property(name="counterparty", data_type=DataType.TEXT),
            Property(name="counterparty_lc", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="entities_json", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="thread_summary", data_type=DataType.TEXT),
            Property(name="latest_message", data_type=DataType.TEXT),
            Property(name="participants_text", data_type=DataType.TEXT),
            Property(name="latest_sent_at", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="confidence", data_type=DataType.NUMBER),
            Property(name="archive_recommendation", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="urgency_reason", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="missing_info_json", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="handler_name", data_type=DataType.TEXT),
            Property(name="handler_name_lc", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="handler_email", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="customer_name", data_type=DataType.TEXT),
            Property(name="customer_name_lc", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="customer_email", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="user_email", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="user_email_lc", data_type=DataType.TEXT, skip_vectorization=True),
        ]

        if THREAD_CLASS not in existing:
            client.collections.create(
                name=THREAD_CLASS,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                properties=thread_props,
            )
        else:
            col = client.collections.get(THREAD_CLASS)
            existing_props = {p.name for p in col.config.get().properties}
            for prop in thread_props:
                if prop.name not in existing_props:
                    col.config.add_property(prop)

        detail_props = [
            Property(name="thread_id", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="thread_key", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="thread_ref", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="user_email", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="user_email_lc", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="full_text", data_type=DataType.TEXT),
            Property(name="participants_text", data_type=DataType.TEXT),
            Property(name="messages_json", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="latest_sent_at", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="earliest_sent_at", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="message_count", data_type=DataType.NUMBER, skip_vectorization=True),
        ]

        if DETAIL_CLASS not in existing:
            client.collections.create(
                name=DETAIL_CLASS,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                properties=detail_props,
            )
        else:
            col = client.collections.get(DETAIL_CLASS)
            existing_props = {p.name for p in col.config.get().properties}
            for prop in detail_props:
                if prop.name not in existing_props:
                    col.config.add_property(prop)

        summary_props = [
            Property(name="day", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="user_email", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="user_email_lc", data_type=DataType.TEXT, skip_vectorization=True),
            Property(name="summary_md", data_type=DataType.TEXT),
            Property(name="total_threads", data_type=DataType.NUMBER, skip_vectorization=True),
            Property(name="action_required", data_type=DataType.NUMBER, skip_vectorization=True),
            Property(name="informational", data_type=DataType.NUMBER, skip_vectorization=True),
            Property(name="irrelevant", data_type=DataType.NUMBER, skip_vectorization=True),
            Property(name="created_at", data_type=DataType.TEXT, skip_vectorization=True),
        ]

        if DAILY_SUMMARY_CLASS not in existing:
            client.collections.create(
                name=DAILY_SUMMARY_CLASS,
                vectorizer_config=Configure.Vectorizer.text2vec_openai(),
                properties=summary_props,
            )
        else:
            col = client.collections.get(DAILY_SUMMARY_CLASS)
            existing_props = {p.name for p in col.config.get().properties}
            for prop in summary_props:
                if prop.name not in existing_props:
                    col.config.add_property(prop)
    finally:
        client.close()
