<template>
  <div class="monitor-page">
    <!-- 顶部导航 -->
    <div class="top-bar">
      <div class="nav-left">
        <button class="back-btn" @click="$router.push('/')">← 返回</button>
        <span class="title">📹 课堂实时监测</span>
      </div>
      <div class="nav-right">
        <span class="class-tag">{{ className || '未选择班级' }}</span>
        <span class="recording-indicator" :class="{ active: recording }">
          {{ recording ? '● REC' : '● STANDBY' }}
        </span>
      </div>
    </div>

    <!-- 控制栏 -->
    <div class="control-bar">
      <button class="btn-start" @click="startRecord" :disabled="recording">
        {{ recording ? '录制中...' : '▶️ 开始录制 & 分析' }}
      </button>
      <button class="btn-stop" @click="stopRecord" :disabled="!recording">
        ⏹ 停止并保存
      </button>

      <div class="model-select">
        <label>模型：</label>
        <select v-model="selectedModel" @change="changeModel" :disabled="recording">
          <option v-for="m in modelOptions" :key="m" :value="m">{{ m }}</option>
        </select>
      </div>

      <div class="timer" v-if="recording">
        ⏱️ {{ formatDuration(recordDuration) }}
      </div>
    </div>

    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 左侧：视频流 -->
      <div class="video-section">
        <div class="video-frame">
          <img
            :src="videoFeedUrl"
            class="video-feed"
            @error="videoError = true"
            @load="videoError = false"
          />
          <div v-if="videoError" class="video-error">
            <p>📷 摄像头未连接</p>
            <p>请确认模型推理服务已启动</p>
          </div>
          <div v-if="recording" class="rec-badge">REC {{ formatDuration(recordDuration) }}</div>
        </div>
      </div>

      <!-- 右侧：实时数据 -->
      <div class="data-section">
        <!-- 实时统计 -->
        <div class="stats-card">
          <h3>📊 实时行为统计</h3>
          <div class="stat-grid">
            <div class="stat-item">
              <span class="stat-emoji">🙋‍♂️</span>
              <div class="stat-info">
                <p class="stat-val">{{ stats.hand_up }}</p>
                <p class="stat-name">举手</p>
              </div>
            </div>
            <div class="stat-item">
              <span class="stat-emoji">📖</span>
              <div class="stat-info">
                <p class="stat-val">{{ stats.study_norm }}</p>
                <p class="stat-name">正常学习</p>
              </div>
            </div>
            <div class="stat-item">
              <span class="stat-emoji">🙅‍♂️</span>
              <div class="stat-info">
                <p class="stat-val">{{ stats.look_down }}</p>
                <p class="stat-name">低头</p>
              </div>
            </div>
            <div class="stat-item">
              <span class="stat-emoji">🚫</span>
              <div class="stat-info">
                <p class="stat-val">{{ stats.abnormal }}</p>
                <p class="stat-name">手机/睡觉</p>
              </div>
            </div>
          </div>

          <!-- 专注度 -->
          <div class="focus-bar">
            <div class="focus-label">
              <span>🎯 课堂专注度</span>
              <span class="focus-value" :class="focusClass">{{ focusRate }}%</span>
            </div>
            <div class="progress-track">
              <div class="progress-fill" :class="focusClass" :style="{ width: focusRate + '%' }"></div>
            </div>
          </div>
        </div>

        <!-- 实时日志 -->
        <div class="log-card">
          <h3>📋 实时日志 <span class="log-count">{{ logs.length }} 条</span></h3>
          <div class="log-scroll" ref="logContainer">
            <div v-if="logs.length === 0" class="log-empty">等待录制开始...</div>
            <div v-for="(log, idx) in logs" :key="idx" class="log-item">
              {{ log }}
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import axios from 'axios'

const modelOptions = ref([])
const selectedModel = ref('')
const recording = ref(false)
const videoError = ref(false)
const recordDuration = ref(0)
const logContainer = ref(null)

const className = ref('')
const teacherCode = ref('')

const videoFeedUrl = ref('http://localhost:5002/video_feed')

const stats = ref({
  hand_up: 0,
  study_norm: 0,
  look_down: 0,
  abnormal: 0
})
const logs = ref([])
const focusRate = ref(0)

const focusClass = computed(() => {
  if (focusRate.value >= 85) return 'good'
  if (focusRate.value >= 60) return 'medium'
  return 'bad'
})

let pollTimer = null
let durationTimer = null

const formatDuration = (seconds) => {
  const m = Math.floor(seconds / 60).toString().padStart(2, '0')
  const s = (seconds % 60).toString().padStart(2, '0')
  return `${m}:${s}`
}

// 自动滚动日志
watch(logs, () => {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
}, { deep: true })

// 拉取实时数据
async function fetchRealtime() {
  try {
    const res = await axios.get('http://localhost:5002/get_realtime_stats')
    stats.value = res.data.stats
    logs.value = res.data.logs || []
    focusRate.value = res.data.focus_rate
  } catch {
    // silent
  }
}

// 录制状态
async function fetchStatus() {
  try {
    const res = await axios.get('http://localhost:5002/get_record_status')
    recording.value = res.data.recording
  } catch {
    // silent
  }
}

// 开始录制
async function startRecord() {
  try {
    await axios.post('http://localhost:5002/start_record', {
      teacher_code: teacherCode.value || 'T2025001',
      class_code: 1,
      lesson_section: '实时课堂'
    })
    recording.value = true
    recordDuration.value = 0

    // 开始轮询
    pollTimer = setInterval(fetchRealtime, 800)
    durationTimer = setInterval(() => { recordDuration.value++ }, 1000)
  } catch (err) {
    alert('启动录制失败，请确认服务已连接')
  }
}

// 停止录制
async function stopRecord() {
  try {
    await axios.post('http://localhost:5002/stop_record', {
      teacher_code: teacherCode.value || 'T2025001',
      class_code: 1,
      lesson_section: '实时课堂'
    })
    recording.value = false

    if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
    if (durationTimer) { clearInterval(durationTimer); durationTimer = null }

    alert('✅ 课堂录制已保存！可在教师管理后台查看报告。')
  } catch (err) {
    alert('停止录制失败')
  }
}

// 模型管理
async function fetchModels() {
  try {
    const r = await axios.get('http://localhost:5002/get_models')
    modelOptions.value = r.data.models
    selectedModel.value = r.data.current
  } catch {
    // silent
  }
}

async function changeModel() {
  await axios.post('http://localhost:5002/switch_model', {
    model_name: selectedModel.value
  })
}

onMounted(() => {
  const user = JSON.parse(localStorage.getItem('terminalUser'))
  if (user) {
    teacherCode.value = user.teacher_code || ''
    // 加载班级信息
    axios.post('http://localhost:5001/teacher-class', { user_code: user.user_code })
      .then(({ data }) => { className.value = data.class_name })
      .catch(() => {})
  }

  fetchModels()
  fetchStatus()
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
  if (durationTimer) clearInterval(durationTimer)
})
</script>

<style scoped>
.monitor-page {
  background: #0a0a0a;
  color: #e0e0e0;
  min-height: 100vh;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

/* 顶部导航 */
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 24px;
  background: #111;
  border-bottom: 1px solid #1a3a1a;
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 16px;
}

.back-btn {
  background: transparent;
  border: 1px solid #333;
  color: #888;
  padding: 6px 12px;
  border-radius: 4px;
  cursor: pointer;
}

.back-btn:hover { border-color: #4ade80; color: #4ade80; }

.title {
  font-size: 16px;
  font-weight: 700;
  color: #fff;
}

.nav-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

.class-tag {
  background: #1a2a1a;
  border: 1px solid #2a4a2a;
  padding: 4px 12px;
  border-radius: 20px;
  font-size: 12px;
  color: #4ade80;
}

.recording-indicator {
  font-size: 12px;
  color: #555;
}

.recording-indicator.active {
  color: #f44;
  animation: blink 1s infinite;
}

@keyframes blink {
  50% { opacity: 0.3; }
}

/* 控制栏 */
.control-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 24px;
  background: #111;
  border-bottom: 1px solid #1a1a1a;
}

.btn-start {
  background: #166534;
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}

.btn-start:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-start:not(:disabled):hover { background: #15803d; }

.btn-stop {
  background: #991b1b;
  color: #fff;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  cursor: pointer;
  font-weight: 600;
}

.btn-stop:disabled { opacity: 0.5; cursor: not-allowed; }
.btn-stop:not(:disabled):hover { background: #b91c1c; }

.model-select {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: #888;
}

.model-select select {
  background: #1a1a1a;
  border: 1px solid #333;
  color: #ccc;
  padding: 6px 10px;
  border-radius: 4px;
}

.timer {
  font-size: 18px;
  color: #f44;
  font-weight: 700;
  margin-left: 12px;
}

/* 主内容区 */
.main-content {
  display: flex;
  gap: 20px;
  padding: 20px 24px;
}

/* 视频流 */
.video-section {
  width: 50%;
}

.video-frame {
  position: relative;
  background: #111;
  border: 2px solid #1a3a1a;
  border-radius: 8px;
  overflow: hidden;
  aspect-ratio: 4/3;
}

.video-feed {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.video-error {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #111;
  color: #555;
}

.video-error p { margin: 4px 0; }

.rec-badge {
  position: absolute;
  top: 12px;
  right: 12px;
  background: rgba(244, 63, 94, 0.8);
  color: #fff;
  padding: 4px 10px;
  border-radius: 4px;
  font-size: 12px;
  font-weight: 700;
  animation: blink 1.5s infinite;
}

/* 数据区 */
.data-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.stats-card {
  background: #111;
  border: 1px solid #222;
  border-radius: 8px;
  padding: 16px;
}

.stats-card h3 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #fff;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 10px;
  margin-bottom: 16px;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 10px;
  background: #0a0a0a;
  border-radius: 6px;
  padding: 10px;
}

.stat-emoji { font-size: 20px; }

.stat-val {
  margin: 0;
  font-size: 22px;
  font-weight: 700;
  color: #fff;
}

.stat-name {
  margin: 0;
  font-size: 11px;
  color: #666;
}

/* 专注度 */
.focus-bar { margin-top: 8px; }

.focus-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
  font-size: 13px;
  color: #aaa;
}

.focus-value {
  font-weight: 700;
  font-size: 20px;
}

.focus-value.good { color: #4ade80; }
.focus-value.medium { color: #f59e0b; }
.focus-value.bad { color: #f44; }

.progress-track {
  height: 8px;
  background: #1a1a1a;
  border-radius: 4px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.5s ease;
}

.progress-fill.good { background: linear-gradient(90deg, #22c55e, #4ade80); }
.progress-fill.medium { background: linear-gradient(90deg, #d97706, #f59e0b); }
.progress-fill.bad { background: linear-gradient(90deg, #dc2626, #f44); }

/* 日志 */
.log-card {
  background: #111;
  border: 1px solid #222;
  border-radius: 8px;
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.log-card h3 {
  margin: 0 0 10px 0;
  font-size: 14px;
  color: #fff;
  display: flex;
  align-items: center;
  gap: 8px;
}

.log-count {
  font-size: 11px;
  color: #555;
  font-weight: normal;
}

.log-scroll {
  flex: 1;
  max-height: 300px;
  overflow-y: auto;
  background: #0a0a0a;
  border: 1px solid #1a1a1a;
  border-radius: 4px;
  padding: 10px;
}

.log-empty {
  color: #333;
  text-align: center;
  padding: 20px;
}

.log-item {
  padding: 4px 0;
  font-size: 12px;
  color: #4ade80;
  border-bottom: 1px solid #111;
}
</style>