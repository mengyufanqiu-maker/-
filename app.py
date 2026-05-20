# -*- coding: utf-8 -*-
"""
PharmaCheck AI — 临床用药风控智能体
Flask 后端：SSE 流式输出 + DeepSeek API
"""

import base64
import json
import os
import mimetypes
import socket

import requests
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, stream_with_context
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL_TEXT = "deepseek-chat"
DEEPSEEK_MODEL_VISION = "deepseek-chat"

SYSTEM_PROMPT = """你是一位任职于国内三甲医院的**主管临床药师**，拥有深厚的临床药物治疗学与药物警戒专业背景。你的职责是对患者拟用或正在使用的药物方案进行**严厉、严谨**的安全审查，绝不姑息潜在用药风险。

## 核心审查维度（必须逐项覆盖）

### 1. 细胞色素 P450 (CYP) 酶系竞争
- 逐一分析涉及药物经 **CYP1A2、2C9、2C19、2D6、3A4** 等亚型的代谢途径。
- 明确标注：底物 / 抑制剂 / 诱导剂 关系，是否存在**竞争性抑制**或**代谢通路饱和**。
- 评估相互作用导致的血药浓度升高（中毒）或降低（疗效不足）风险，给出**临床处置建议**。

### 2. 肝肾毒性叠加
- 识别具有**肝毒性**（肝酶升高、肝衰竭）或**肾毒性**（肾小管损伤、肾功能下降）的药物。
- 评估多药联用时的**毒性叠加**风险，特别关注老年、脱水、合并 CKD/肝硬化患者。
- 标注需监测的实验室指标（ALT、AST、肌酐、eGFR 等）及复查周期。

### 3. 非甾体抗炎药 (NSAID) 出血风险
- 若方案含 NSAID（如布洛芬、双氯芬酸、塞来昔布、阿司匹林等），**必须**评估：
  - 消化道出血、溃疡风险
  - 与抗凝/抗血小板药（华法林、利伐沙班、氯吡格雷等）的协同出血风险
  - 肾功能影响及心血管风险

## 输出要求
- **必须使用 Markdown 格式**输出，结构清晰、层次分明。
- 建议结构：
  - `# 临床用药安全审查报告`
  - `## 患者用药清单摘要`
  - `## CYP 酶系相互作用分析`
  - `## 肝肾毒性叠加评估`
  - `## NSAID 相关出血风险`
  - `## 综合风险等级与处置建议`（用 **高/中/低** 标注）
- 语气专业、严谨、直接；对高风险组合须用**加粗**或引用块明确警示。
- 若信息不足，列出需补充的关键信息，但仍基于已有信息给出最大可能的风险提示。

## 免责声明（报告末尾必须附上）
> ⚠️ 本报告由 AI 辅助生成，仅供学术与教学参考，**不能替代**执业医师/药师的临床决策。实际用药请严格遵医嘱。"""


def _sse_payload(content: str, error: bool = False) -> str:
    return f"data: {json.dumps({'content': content, 'error': error}, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


def _build_user_message(check_mode: str, text_content: str, file_storage) -> list:
    """构建 DeepSeek messages 中的 user 部分（支持多模态）。"""
    if check_mode == "image":
        if not file_storage or not file_storage.filename:
            return [{"role": "user", "content": "【错误】未收到有效的处方/药品照片，请重新上传。"}]

        raw = file_storage.read()
        if not raw:
            return [{"role": "user", "content": "【错误】上传的文件为空，请重新选择照片。"}]

        mime = file_storage.mimetype or mimetypes.guess_type(file_storage.filename)[0] or "image/jpeg"
        b64 = base64.b64encode(raw).decode("utf-8")
        data_url = f"data:{mime};base64,{b64}"

        prompt = (
            "请根据上传的处方或药品包装照片，识别其中所有可见药物名称、剂量与用法，"
            "并依据主管临床药师标准完成用药安全审查（CYP 酶竞争、肝肾毒性叠加、NSAID 出血风险）。"
        )
        if text_content and text_content.strip():
            prompt += f"\n\n用户补充说明：{text_content.strip()}"

        return [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]

    drugs = (text_content or "").strip()
    if not drugs:
        return [{"role": "user", "content": "【错误】未提供任何药物名称，请输入待审查的药品清单。"}]

    return [
        {
            "role": "user",
            "content": (
                f"请对以下药物/用药方案进行严格的临床用药安全审查：\n\n{drugs}\n\n"
                "请按系统要求，从 CYP 酶竞争、肝肾毒性叠加、NSAID 出血风险三个维度给出 Markdown 格式报告。"
            ),
        }
    ]


def _stream_deepseek(api_key: str, messages: list, model: str):
    """调用 DeepSeek Chat Completions 并以 SSE 形式逐块 yield。"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "stream": True,
    }

    try:
        resp = requests.post(
            DEEPSEEK_API_URL,
            headers=headers,
            json=body,
            stream=True,
            timeout=120,
        )
    except requests.RequestException as exc:
        yield _sse_payload(
            f'<span style="color:#dc2626;font-weight:bold;">网络请求失败：{exc}</span>',
            error=True,
        )
        yield _sse_done()
        return

    if resp.status_code != 200:
        try:
            err_body = resp.json()
            detail = err_body.get("error", {}).get("message", resp.text)
        except Exception:
            detail = resp.text[:500]
        yield _sse_payload(
            f'<span style="color:#dc2626;font-weight:bold;">DeepSeek API 错误 ({resp.status_code})：{detail}</span>',
            error=True,
        )
        yield _sse_done()
        return

    for raw_line in resp.iter_lines(decode_unicode=True):
        if not raw_line:
            continue
        line = raw_line.strip()
        if not line.startswith("data:"):
            continue
        data_str = line[5:].strip()
        if data_str == "[DONE]":
            break
        try:
            chunk = json.loads(data_str)
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            piece = delta.get("content", "")
            if piece:
                yield _sse_payload(piece, error=False)
        except json.JSONDecodeError:
            continue

    yield _sse_done()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/check_drugs", methods=["POST"])
def check_drugs():
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not api_key:
        def missing_key_stream():
            yield _sse_payload(
                '<span style="color:#dc2626;font-weight:bold;">'
                "❌ 错误：未检测到环境变量 DEEPSEEK_API_KEY。"
                "请在操作系统或 .env 中配置有效的 DeepSeek API 密钥后重启本服务。"
                "</span>",
                error=True,
            )
            yield _sse_done()

        return Response(
            stream_with_context(missing_key_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    check_mode = request.form.get("check_mode", "text").strip().lower()
    if check_mode not in ("text", "image"):
        check_mode = "text"

    text_content = request.form.get("text_content", "")
    file_content = request.files.get("file_content")

    user_msgs = _build_user_message(check_mode, text_content, file_content)
    model = DEEPSEEK_MODEL_VISION if check_mode == "image" else DEEPSEEK_MODEL_TEXT

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if isinstance(user_msgs, list) and user_msgs and user_msgs[0].get("role") == "user":
        messages.extend(user_msgs)
    else:
        messages.append({"role": "user", "content": str(user_msgs)})

    # 本地校验错误（无药名/无文件）直接 SSE 返回
    first_content = messages[-1].get("content", "")
    if isinstance(first_content, str) and first_content.startswith("【错误】"):

        def local_err_stream():
            yield _sse_payload(
                f'<span style="color:#dc2626;font-weight:bold;">{first_content}</span>',
                error=True,
            )
            yield _sse_done()

        return Response(
            stream_with_context(local_err_stream()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    return Response(
        stream_with_context(_stream_deepseek(api_key, messages, model)),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


def _patch_socket_getfqdn():
    """
    Windows 计算机名为中文时，http.server 绑定端口会调用 socket.getfqdn，
    反向解析可能触发 UnicodeDecodeError。对本地地址直接返回 localhost。
    """
    _orig_getfqdn = socket.getfqdn
    _local_hosts = frozenset(("127.0.0.1", "localhost", "::1", "0.0.0.0", "[::1]"))

    def _safe_getfqdn(name=""):
        if not name or name in _local_hosts:
            return "localhost"
        try:
            return _orig_getfqdn(name)
        except UnicodeDecodeError:
            return "localhost"

    socket.getfqdn = _safe_getfqdn


if __name__ == "__main__":
    _patch_socket_getfqdn()
    app.run(host="127.0.0.1", port=5003, debug=True, threaded=True)
