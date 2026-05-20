import os
import requests
import json
from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/check_drugs', methods=['POST'])
def check_drugs():
    # 从 Render 环境变量读取 Key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not api_key:
        return jsonify({"error": "云端未配置 DEEPSEEK_API_KEY"}), 500

    text_content = request.form.get("text_content", "").strip()
    if not text_content:
        return jsonify({"error": "未提供药物名称"}), 400

    # 极简且极其严格的临床药师 Prompt
    system_prompt = f"""你是一位拥有 15 年三甲医院经验的主管临床药师。
请对以下患者用药清单进行严格审查：【{text_content}】

要求：
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

    def generate_stream():
        try:
            resp = requests.post(
                "https://api.deepseek.com/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": system_prompt}],
                    "temperature": 0.0,
                    "stream": True,
                },
                stream=True,
                timeout=60
            )

            if resp.status_code != 200:
                yield f"data: {json.dumps({'error': f'DeepSeek 接口报错: {resp.status_code}'})}\n\n"
                return

            for line in resp.iter_lines():
                if line:
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith('data: '):
                        data_str = decoded[6:]
                        if data_str == '[DONE]':
                            break
                        try:
                            chunk = json.loads(data_str)
                            content = chunk['choices'][0]['delta'].get('content', '')
                            if content:
                                yield f"data: {json.dumps({'text': content})}\n\n"
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            yield f"data: {json.dumps({'error': f'服务器网络异常: {str(e)}'})}\n\n"

    return Response(generate_stream(), mimetype='text/event-stream')

if __name__ == '__main__':
    # 动态适配云端端口，完美兼容 Render
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port)
