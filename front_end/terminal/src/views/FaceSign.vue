<template>
  <div class="face-sign-page">
    <!-- 顶部导航 -->
    <div class="top-bar">
      <div class="nav-left">
        <button class="back-btn" @click="$router.push('/')">← 返回</button>
        <span class="title">📷 人脸识别签到</span>
      </div>
      <div class="nav-right">
        <span class="class-tag">{{ className || '未选择班级' }}</span>
        <span class="service-status">
          <span class="status-dot" :class="{ online: serviceUp }"></span>
          {{ serviceUp ? '服务在线' : '服务离线' }}
        </span>
      </div>
    </div>

    <!-- 主内容区 -->
    <div class="main-content">
      <!-- 左侧：视频流 -->
      <div class="video-section">
        <div class="video-frame">
          <img
            :src="videoFeedUrl"
            class="live-video"
            @error="videoError = true"
            @load="videoError = false"
          />
          <div v-if="videoError" class="video-error">
            <p>📷 人脸签到服务未连接</p>
            <p>请确认 face-recognition-service 已启动（端口 5003）</p>
          </div>
          <div class="video-overlay" v-if="!videoError">
            <span class="overlay-text">实时人脸识别中</span>
          </div>
        </div>

        <!-- 签到统计 -->
        <div class="sign-stats">
          <div class="sign-stat-item">
            <span class="sign-stat-val">{{ signCount }}</span>
            <span class="sign-stat-label">已签到</span>
          </div>
          <div class="sign-stat-item">
            <span class="sign-stat-val">{{ totalStudents }}</span>
            <span class="sign-stat-label">应到人数</span>
          </div>
          <div class="sign-stat-item">
            <span class="sign-stat-val" :class="{ good: signRate >= 80, bad: signRate < 50 }">
              {{ signRate }}%
            </span>
            <span class="sign-stat-label">签到率</span>
          </div>
        </div>
      </div>

      <!-- 右侧：签到记录 -->
      <div class="record-section">
        <div class="record-card">
          <div class="record-header">
            <h3>📋 签到记录</h3>
            <button class="refresh-btn" @click="fetchLogs">🔄 刷新</button>
          </div>

          <div class="record-scroll" ref="logContainer">
            <div v-if="logs.length === 0" class="record-empty">
              等待签到记录...
            </div>
            <div
              v-for="(log, i) in logs"
              :key="i"
              class="record-item"
              :class="{ 'is-new': i >= logs.length - signCount }"
            >
              <span class="record-dot"></span>
              <span class="record-text">{{ formatLog(log) }}</span>
            </div>
          </div>
        </div>

        <!-- 快捷操作 -->
        <div class="action-bar">
          <p class="action-hint">
            💡 签到说明：学生面对摄像头，系统自动识别并记录签到时间
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted, nextTick, watch } from 'vue'
import axios from 'axios'

const videoFeedUrl = ref('http://localhost:5003/video_feed')
const videoError = ref(false)
const serviceUp = ref(false)
const className = ref('')
const totalStudents = ref(0)
const logContainer = ref(null)

const logs = ref([])

// 签到人数（从日志中解析唯一姓名）
const signCount = computed(() => {
  const names = new Set()
  logs.value.forEach(log => {
    const match = log.trim().match(/^(.+?)\s+\d{4}-/)
    if (match) names.add(match[1])
  })
  return names.size
})

const signRate = computed(() => {
  if (totalStudents.value === 0) return 0
  return Math.round((signCount.value / totalStudents.value) * 100)
})

const formatLog = (log) => {
  return log.trim()
}

// 自动滚动
watch(logs, () => {
  nextTick(() => {
    if (logContainer.value) {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    }
  })
}, { deep: true })

let pollTimer = null

async function fetchLogs() {
  try {
    const res = await axios.get('http://localhost:5003/get_sign_log')
    logs.value = res.data.logs || []
  } catch {
    // silent
  }
}

async function checkService() {
  try {
    await axios.get('http://localhost:5003/', { timeout: 2000 })
    serviceUp.value = true
  } catch {
    serviceUp.value = false
  }
}

async function loadClassInfo() {
  const user = JSON.parse(localStorage.getItem('terminalUser'))
  if (!user) return

  try {
    const { data } = await axios.post('http://localhost:5001/teacher-class', {
      user_code: user.user_code
    })
    className.value = data.class_name
    totalStudents.value = data.student_count || 0
  } catch {
    // silent
  }
}

onMounted(() => {
  loadClassInfo()
  checkService()
  fetchLogs()
  pollTimer = setInterval(() => {
    checkService()
    fetchLogs()
  }, 2000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.face-sign-page {
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

.service-status {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #888;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #555;
}

.status-dot.online {
  background: #4ade80;
  box-shadow: 0 0 6px #4ade80;
}

/* 主内容区 */
.main-content {
  display: flex;
  gap: 20px;
  padding: 20px 24px;
}

/* 视频区 */
.video-section {
  width: 55%;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.video-frame {
  position: relative;
  background: #111;
  border: 2px solid #1a3a1a;
  border-radius: 8px;
  overflow: hidden;
  aspect-ratio: 4/3;
}

.live-video {
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

.video-overlay {
  position: absolute;
  bottom: 12px;
  left: 12px;
  background: rgba(0, 0, 0, 0.6);
  padding: 4px 10px;
  border-radius: 4px;
}

.overlay-text {
  font-size: 12px;
  color: #4ade80;
}

/* 签到统计 */
.sign-stats {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 12px;
}

.sign-stat-item {
  background: #111;
  border: 1px solid #222;
  border-radius: 8px;
  padding: 16px;
  text-align: center;
}

.sign-stat-val {
  display: block;
  font-size: 28px;
  font-weight: 700;
  color: #fff;
}

.sign-stat-val.good { color: #4ade80; }
.sign-stat-val.bad { color: #f44; }

.sign-stat-label {
  display: block;
  font-size: 12px;
  color: #666;
  margin-top: 4px;
}

/* 记录区 */
.record-section {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.record-card {
  background: #111;
  border: 1px solid #222;
  border-radius: 8px;
  padding: 16px;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.record-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.record-header h3 {
  margin: 0;
  font-size: 14px;
  color: #fff;
}

.refresh-btn {
  background: transparent;
  border: 1px solid #333;
  color: #888;
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.refresh-btn:hover { border-color: #4ade80; color: #4ade80; }

.record-scroll {
  flex: 1;
  max-height: 500px;
  overflow-y: auto;
  background: #0a0a0a;
  border: 1px solid #1a1a1a;
  border-radius: 4px;
  padding: 10px;
}

.record-empty {
  color: #333;
  text-align: center;
  padding: 30px;
}

.record-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  border-bottom: 1px solid #111;
  font-size: 13px;
  color: #aaa;
  transition: color 0.3s;
}

.record-item.is-new {
  color: #4ade80;
}

.record-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #333;
  flex-shrink: 0;
}

.record-item.is-new .record-dot {
  background: #4ade80;
  box-shadow: 0 0 4px #4ade80;
}

.record-text {
  flex: 1;
  word-break: break-all;
}

/* 底部操作 */
.action-bar {
  background: #111;
  border: 1px solid #222;
  border-radius: 8px;
  padding: 16px;
}

.action-hint {
  margin: 0;
  font-size: 12px;
  color: #555;
  line-height: 1.6;
}
</style>