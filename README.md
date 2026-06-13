# Field Event Visualization Demos v2.0-qwen-llm

本版本在原始六个 no-leak 可视化 demo 基础上增加了 Qwen 大模型高层判读，并把五个物理场的色条改为低值蓝色、高值红色。

## 六个 demo

1. `demo_01_fire.py`：着火 / 明火初期
2. `demo_02_electrical_overheat.py`：电气柜过热
3. `demo_03_water_leak.py`：漏水
4. `demo_04_steam_leak.py`：蒸汽泄漏
5. `demo_05_co2_ventilation.py`：CO2 积聚 / 通风不良
6. `demo_06_dust_pollution.py`：空气污染 / 粉尘泄漏

也可以直接运行统一入口：

```bash
python field_event_visual_demos.py --demo fire
python field_event_visual_demos.py --demo electrical
python field_event_visual_demos.py --demo leak
python field_event_visual_demos.py --demo steam
python field_event_visual_demos.py --demo co2
python field_event_visual_demos.py --demo dust
```

## 大模型高层判读

默认模型名：`qwen3.6-flash`
默认触发频率：每 50 步一次
默认接口：DashScope OpenAI-compatible Chat Completions endpoint

为了避免泄露密钥，本代码包不会硬编码 API Key。推荐使用环境变量：

```bash
export DASHSCOPE_API_KEY="你的 DashScope / Qwen API Key"
$env:DASHSCOPE_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
python field_event_visual_demos.py --demo fire
```

也可以显式指定：

```bash
python field_event_visual_demos.py --demo fire --llm-model qwen3.6-flash --llm-interval-steps 50
```

禁用大模型调用：

```bash
python field_event_visual_demos.py --demo fire --disable-llm
```

界面逻辑：

- 第 50、100、150、200 ... 步，将当前 `field_event` 编码器输出的事件列表发送给大模型。
- 大模型未返回前，界面显示：`高层判读：大模型判读中...`
- 大模型返回后，界面显示：
  - `推断的异常事件`
  - `简短的推断理由或者过程`
- 大模型输入只包含编码器事件列表，不包含正常背景、注入异常、真值标签、mask、异常中心或事故类型。

## 结构化 JSON 输出

提示词要求大模型必须输出合法 JSON：

```json
{
  "推断的异常事件": "...",
  "简短的推断理由或者过程": "..."
}
```

## 大模型准确性评估

可以在无 GUI 模式下对六个场景各调用一次 Qwen，并计算简单的场景级匹配率：

```bash
export DASHSCOPE_API_KEY="你的 DashScope / Qwen API Key"
python field_event_visual_demos.py --llm-eval --llm-eval-step 150 --llm-eval-out qwen_eval_results.json
```

说明：当前评估采用场景名称关键字匹配，适合快速检查 demo 层大模型判读是否符合预期。论文实验可以把该脚本扩展为多 seed、多时间点、多模型对照。

## 核心 no-leak 约束

- 地图初始化为 30×30，障碍物密度 0.2，并使用 `make_obstacle_grid` 填掉小型不连通自由区域。
- 每个 demo 默认运行 200 步。
- 每一步渲染五个物理场：温度、湿度、气压/压差、CO2、空气质量。
- 编码器只接收 `encoder.update(t, current_fields)`，其中 `current_fields` 是背景和注入异常混合后的五个场。
- 正常背景、异常注入、事件类型、mask、中心点等不会作为单独通道或参数输入编码器。
- Matplotlib 图像对象只初始化一次，动画中只调用 `set_data` 和文本更新，避免每步重建图形导致卡顿。
- 自动选择 Noto/思源/微软雅黑/黑体/WenQuanYi 等 CJK 字体，避免中文显示成方块。

## 无 GUI 快速检查

```bash
python field_event_visual_demos.py --headless-check
```
