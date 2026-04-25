<template>
  <div class="monitor-page">
    <h1>>> 📊 课堂实时监测看板</h1>

    <div style="margin-bottom:16px">
      <button @click="startRecord" :disabled="recording" style="padding:10px 20px;margin-right:10px">
        🎥 开始录制 & 分析
      </button>
      <button @click="stopRecord" :disabled="!recording" style="padding:10px 20px">
        ⏹ 停止并保存
      </button>
      <span style="color:#0f0;margin-left:16px">
        状态：{{ recording ? '🟢 录制中' : '⚫ 待机' }}
      </span>
    </div>

    <div class="split-container">
      <div class="left-section">
        <div class="box">
          <h3>> 实时画面</h3>
          <div class="video-phone-frame">
            <img src="http://localhost:5002/video_feed" class="video-feed" />
          </div>
        </div>

        <div class="box">
          <h3>> 模型配置</h3>
          <div class="row">
            <label>当前模型：</label>
            <select v-model="selectedModel" @change="changeModel">
              <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
            </select>
          </div>
        </div>
      </div>

      <div class="right-section">
        <div class="box">
          <h3>> 行为统计（实时）</h3>
          <div class="stat-grid">
            <div>
              <label>🙋‍♂️ 举手</label>
              <span>{{ stats.hand_up }}</span>
            </div>
            <div>
              <label>📖 正常学习</label>
              <span>{{ stats.study_norm }}</span>
            </div>
            <div>
              <label>🙅‍♂️ 低头</label>
              <span>{{ stats.look_down }}</span>
            </div>
            <div>
              <label>🚫 手机/睡觉</label>
              <span>{{ stats.abnormal }}</span>
            </div>
            <div>
              <label>📈 专注度</label>
              <span class="green">{{ focusRate }}%</span>
            </div>
          </div>
        </div>

        <div class="box log-box">
          <h3>> 实时日志</h3>
          <div class="logs">
            <p v-for="(log, idx) in logs" :key="idx" class="log-item">{{ log }}</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const modelOptions = ref([])
const selectedModel = ref('')
const recording = ref(false)

const stats = ref({
  hand_up: 0,
  study_norm: 0,
  look_down: 0,
  abnormal: 0
})
const logs = ref([])
const focusRate = ref(0)
// 拉取实时数据
async function fetchRealtime() {
  const res = await axios.get('http://localhost:5002/get_realtime_stats')
  stats.value = res.data.stats
  logs.value = res.data.logs
  focusRate.value = res.data.focus_rate   // 👈 就加这一行！
}

// 状态
async function fetchStatus() {
  const res = await axios.get('http://localhost:5002/get_record_status')
  recording.value = res.data.recording
}

// 开始
async function startRecord() {
  await axios.post('http://localhost:5002/start_record', {
    teacher_code: 'T2025001',
    class_id: 1,
    lesson_section: '实时课堂'
  })
  recording.value = true
  setInterval(fetchRealtime, 800)
}

// 停止（自动上传）
async function stopRecord() {
  await axios.post('http://localhost:5002/stop_record', {
    teacher_code: 'T2025001',
    class_id: 1,
    lesson_section: '实时课堂'
  })
  recording.value = false
  alert('已保存到数据库 & MinIO！')
}

// 模型
async function fetchModels() {
  const r = await axios.get('http://localhost:5002/get_models')
  modelOptions.value = r.data.models
  selectedModel.value = r.data.current
}

async function changeModel() {
  await axios.post('http://localhost:5002/switch_model', {
    model_name: selectedModel.value
  })
}

onMounted(() => {
  fetchModels()
  fetchStatus()
})
</script>

<style scoped>
.monitor-page {
  padding: 24px;
  color: #0f0;
  background: #000;
  min-height: 100vh;
}

.split-container {
  display: flex;
  gap: 20px;
  margin-top: 16px;
}

.left-section {
  width: 452px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.right-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.box {
  background: #111;
  border: 1px solid #0f0;
  padding: 16px;
  border-radius: 6px;
}

.video-phone-frame {
  width: 420px;
  height: 236px;
  border: 2px solid #0f0;
  background: #000;
  overflow: hidden;
  position: relative;
}

.video-feed {
  position: absolute;
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.stat-grid>div {
  display: flex;
  justify-content: space-between;
  padding: 6px 0;
}

.logs {
  height: 200px;
  overflow-y: auto;
  background: #000;
  padding: 10px;
  border: 1px solid #333;
}

.log-item {
  margin: 4px 0;
  font-size: 14px;
}
</style>