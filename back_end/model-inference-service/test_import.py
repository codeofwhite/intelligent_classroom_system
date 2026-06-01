import sys
sys.path.insert(0, '.')
try:
    from shared import minio_client, db
    print("shared.py import OK")
except Exception as e:
    print(f"shared.py import FAILED: {e}")

try:
    from chat_agent import chat_agent_api
    print("chat_agent.py import OK")
except Exception as e:
    print(f"chat_agent.py import FAILED: {e}")

try:
    from ai_agent import analyze_class_report
    print("ai_agent.py import OK")
except Exception as e:
    print(f"ai_agent.py import FAILED: {e}")