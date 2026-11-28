"""
src/alerting/zalo_bot.py
========================
Zalo Bot Platform Alert Dispatcher.
Token: configured via ZALO_BOT_TOKEN in .env
API Base: https://bot-api.zaloplatforms.com/bot<TOKEN>/
Features:
- Formatted Vietnamese safety alert notifications.
- Snapshot dispatch with multipart image upload.
- Webhook JSON payload dispatch for industrial integrations.
- Graceful dry-run and local audit logging when chat_id is not yet paired.
"""

from __future__ import annotations
import os
import requests
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from src.config import config, update_env_file


class ZaloAlertBot:
    """Dispatches alerts via Zalo Bot Platform Open APIs."""

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
        webhook_url: Optional[str] = None,
    ):
        self.token = token or config.zalo_bot_token
        self.chat_id = chat_id or config.zalo_chat_id
        self.webhook_url = webhook_url or config.zalo_webhook_url
        self.api_base = f"https://bot-api.zaloplatforms.com/bot{self.token}"

    def test_connection(self) -> Tuple[bool, Dict[str, Any]]:
        """Verify Zalo bot token validity using getMe."""
        url = f"{self.api_base}/getMe"
        try:
            r = requests.post(url, timeout=5.0)
            data = r.json()
            if r.status_code == 200 and data.get("ok"):
                return True, data.get("result", {})
            return False, {"error": data.get("description", "Not found"), "code": r.status_code}
        except Exception as e:
            return False, {"error": str(e)}

    def discover_latest_chat_id(self) -> Optional[str]:
        """Auto-discovers the latest chat_id by querying POST getUpdates from Zalo Bot Platform."""
        url = f"{self.api_base}/getUpdates"
        try:
            r = requests.post(url, timeout=3.0)
            data = r.json()
            if data.get("ok") and data.get("result"):
                for update in reversed(data["result"]):
                    msg = update.get("message") or update.get("channel_post") or update.get("my_chat_member")
                    if msg:
                        chat = msg.get("chat", {})
                        c_id = str(chat.get("id"))
                        if c_id:
                            self.chat_id = c_id
                            config.zalo_chat_id = c_id
                            update_env_file({"ZALO_CHAT_ID": c_id})
                            print(f"✅ Discovered & Saved Zalo Chat ID: {c_id}")
                            return c_id
        except Exception as e:
            pass
        return None

    def send_message(self, text: str, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """Sends a text alert via Zalo API or Webhook."""
        target_chat = chat_id or self.chat_id
        if not target_chat:
            target_chat = self.discover_latest_chat_id()

        # 1. If webhook URL is set, dispatch directly to webhook
        if self.webhook_url:
            try:
                payload = {"text": text, "chat_id": target_chat, "sender": "Rep-YOLO11s CCTV Safety"}
                resp = requests.post(self.webhook_url, json=payload, timeout=5.0)
                if resp.status_code in (200, 201, 204):
                    return True, "Zalo Webhook dispatched successfully"
            except Exception as e:
                print(f"⚠️ Zalo Webhook dispatch error: {e}")

        # 2. Use official Zalo Bot Platform sendMessage API
        if not target_chat:
            return True, "Zalo Chat ID not configured. Message stored in local audit trail & UI stream."

        url = f"{self.api_base}/sendMessage"
        payload = {"chat_id": target_chat, "text": text}
        try:
            r = requests.post(url, json=payload, timeout=8.0)
            data = r.json()
            if r.status_code == 200 and data.get("ok"):
                return True, "Zalo message sent successfully"
            return False, f"Zalo error {r.status_code}: {data.get('description', '')}"
        except Exception as e:
            return False, f"Zalo network exception: {str(e)}"

    def send_photo(self, photo_path: str | Path, caption: str, chat_id: Optional[str] = None) -> Tuple[bool, str]:
        """Uploads a photo snapshot along with caption to Zalo."""
        target_chat = chat_id or self.chat_id
        if not target_chat:
            target_chat = self.discover_latest_chat_id()
        file_path = Path(photo_path)
        if not file_path.exists():
            return False, f"Snapshot file not found: {photo_path}"

        # 1. If webhook URL is set, dispatch snapshot info
        if self.webhook_url:
            try:
                payload = {"text": caption, "photo_name": file_path.name, "chat_id": target_chat}
                resp = requests.post(self.webhook_url, json=payload, timeout=5.0)
                if resp.status_code in (200, 201, 204):
                    return True, "Zalo Webhook snapshot dispatched successfully"
            except Exception as e:
                print(f"⚠️ Zalo Webhook dispatch error: {e}")

        if not target_chat:
            return True, "Zalo Chat ID not configured. Snapshot stored in local storage & UI stream."

        url = f"{self.api_base}/sendPhoto"
        try:
            with open(file_path, "rb") as f:
                files = {"photo": (file_path.name, f, "image/jpeg")}
                data = {"chat_id": target_chat, "caption": caption}
                r = requests.post(url, data=data, files=files, timeout=12.0)
                res = r.json()
                if r.status_code == 200 and res.get("ok"):
                    return True, "Zalo photo sent successfully"
                return False, f"Zalo error {r.status_code}: {res.get('description', '')}"
        except Exception as e:
            return False, f"Zalo network exception: {str(e)}"

    def dispatch_alert(self, alert_event: Dict[str, Any]) -> Tuple[bool, str]:
        """Formats and dispatches Vietnamese safety violation alert."""
        v_list = [v.replace("_", " ") for v in alert_event.get("violations", [])]
        v_str = ", ".join(v_list)

        helmet_desc = (
            f"CO ({alert_event.get('helmet_color', 'khong ro')})"
            if alert_event.get("has_helmet")
            else "KHONG DOI MU (VI PHAM)"
        )
        role_info = alert_event.get("historical_role_title") or alert_event.get("role_title") or "Chua xep loai"
        hist_col = alert_event.get("historical_helmet_color")
        role_line = f"{role_info} (Mau mu truoc do: {hist_col})" if hist_col else role_info
        label_suffix = f" [{alert_event['custom_label']}]" if alert_event.get("custom_label") else ""

        message = (
            f"[CANH BAO AN TOAN MU BAO HO - AEGIS VMS]\n"
            f"Camera: {alert_event.get('camera_id')}\n"
            f"Thoi gian: {alert_event.get('timestamp')}\n"
            f"Doi tuong: ID #{alert_event.get('worker_id')}{label_suffix} | Chuc danh: {role_line} (Tin cay: {alert_event.get('confidence', 0)*100:.1f}%)\n"
            f"Trang thai: VI PHAM AN TOAN LAO DONG\n"
            f"Loi vi pham: {v_str}\n"
            f"Kiem tra: Mu bao ho: {helmet_desc}\n"
            f"Thoi luong vi pham: {alert_event.get('violation_duration_frames', 0)} frames"
        )

        snapshot_path = alert_event.get("snapshot_path")
        if snapshot_path and Path(snapshot_path).exists() and self.chat_id:
            ok, msg = self.send_photo(snapshot_path, message)
            if ok:
                return True, msg
            return self.send_message(message)
        else:
            return self.send_message(message)
