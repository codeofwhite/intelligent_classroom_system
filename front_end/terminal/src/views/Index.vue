<template>
  <div class="console-page">
    <!-- 顶部状态栏 -->
    <div class="top-bar">
      <div class="system-name">🖥️ CLASSROOM TERMINAL</div>
      <div class="user-info">
        <span class="status-dot" :class="{ online: connected }"></span>
        {{ teacherName || '未登录' }}
        <button class="logout-btn" @click="logout">退出</button>
      </div>
    </div>

    <!-- 欢迎区域 -->
    <div class="welcome-section">
      <h1>欢迎回来，{{ teacherName }}</h1>
      <p class="sub-text">{{ todayStr }} · 智慧课堂终端系统</p>
    </div>

    <!-- 班级信息卡片 -->
    <div class="info-grid">
      <div class="info-card blue">
        <div class="card-icon">🏫</div>
        <div class="card-body">
          <p class="card-value">{{ className || '--' }}</p>
          <p class="card-label">当前班级</p>
        </div>
      </div>
      <div class="info-card green">
        <div class="card-icon">📚</div>
        <div class="card-body">
          <p class="card-value">{{ subject || '--' }}</p>
          <p class="card-label">授课科目</p>
        </div>
      </div>
      <div class="info-card purple">
        <div class="card-icon">👥</div>
        <div class="card-body">
          <p class="card-value">{{ studentCount }}</p>
          <p class="card-label">学生总数</p>
        </div>
      </div>
      <div class="info-card orange">
        <div class="card-icon">📊</div>
        <div class="card-body">
          <p class="card-value">{{ reportCount }}</p>
          <p class="card-label">已录制课堂</p>
        </div>
      </div>
    </div>

    <!-- 功能入口 -->
    <div class="action-grid">
      <div class="action-card" @click="$router.push('/face-sign')">
        <div class="action-icon">📷</div>
        <h3>人脸识别签到</h3>
        <p>启动摄像头实时识别学生人脸，自动签到记录</p>
        <div class="action-status">
          <span class="status-dot" :class="{ online: signServiceUp }"></span>
          {{ signServiceUp ? '服务就绪' : '服务未连接' }}
        </div>
      </div>

      <div class="action-card" @click="$router.push('/behavior-monitor')">
        <div class="action-icon">📹</div>
        <h3>课堂实时监测</h3>
        <p>AI 行为分析，实时检测举手、低头、专注度等</p>
        <div class="action-status">
          <span class="status-dot" :class="{ online: monitorServiceUp }"></span>
          {{ monitorServiceUp ? '服务就绪' : '服务未连接' }}
        </div>
      </div>
    </div>

    <!-- 最近签到记录 -->
    <div class="recent-section">
      <h3>📋 最近签到记录</h3>
      <div v-if="recentLogs.length === 0" class="empty">暂无签到记录</div>
      <div v-else class="log-list">
        <div v-for="(log, i) in recentLogs" :key="i" class="log-item">
          <span class="log-dot"></span>
          {{ log.trim() }}
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()

const teacherName = ref('')
const className = ref('')
const subject = ref('')
const studentCount = ref(0)
const reportCount = ref(0)
const connected = ref(false)
const signServiceUp = ref(false)
const monitorServiceUp = ref(false)
const recentLogs = ref([])

const todayStr = new Date().toLocaleDateString('zh-CN', {
  year: 'numeric', month: 'long', day: 'numeric', weekday: 'long'
})

let statusTimer = null

const loadClassInfo = async () => {
  const user = JSON.parse(localStorage.getItem('terminalUser'))
  if (!user) {
    router.push('/login')
    return
  }

  teacherName.value = user.name
  connected.value = true

  try {
    const { data } = await axios.post('http://localhost:5001/teacher-class', {
      user_code: user.user_code
    })
    className.value = data.class_name
    subject.value = data.subject
    studentCount.value = data.student_count
  } catch (err) {
    console.error('加载课室信息失败')
  }
}

const loadReportCount = async () => {
  const user = JSON.parse(localStorage.getItem('terminalUser'))
  if (!user?.teacher_code) return

  try {
    const { data } = await axios.get('http://localhost:5002/api/teacher/reports', {
      params: { teacher_code: user.teacher_code }
    })
    reportCount.value = (data || []).length
  } catch (err) {
    // silent
  }
}

const checkServices = async () => {
  // 检查人脸签到服务
  try {
    await axios.get('http://localhost:5003/', { timeout: 2000 })
    signServiceUp.value = true
  } catch {
    signServiceUp.value = false
  }

  // 检查模型推理服务
  try {
    await axios.get('http://localhost:5002/get_record_status', { timeout: 2000 })
    monitorServiceUp.value = true
  } catch {
    monitorServiceUp.value = false
  }
}

const loadSignLogs = async () => {
  try {
    const { data } = await axios.get('http://localhost:5003/get_sign_log')
    recentLogs.value = (data.logs || []).slice(-5)
  } catch {
    // silent
  }
}

const logout = () => {
  localStorage.removeItem('terminalUser')
  router.push('/login')
}

onMounted(() => {
  loadClassInfo()
  loadReportCount()
  checkServices()
  loadSignLogs()
  statusTimer = setInterval(() => {
    checkServices()
    loadSignLogs()
  }, 10000)
})

onUnmounted(() => {
  if (statusTimer) clearInterval(statusTimer)
})
</script>

<style scoped>
.console-page {
  padding: 0;
  background: #0a0a0a;
  color: #e0e0e0;
  min-height: 100vh;
  font-family: 'SF Mono', 'Fira Code', monospace;
}

/* 顶部状态栏 */
.top-bar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 24px;
  background: #111;
  border-bottom: 1px solid #1a3a1a;
}

.system-name {
  font-size: 16px;
  font-weight: 700;
  color: #4ade80;
  letter-spacing: 2px;
}

.user-info {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 14px;
}

.logout-btn {
  background: transparent;
  border: 1px solid #555;
  color: #999;
  padding: 4px 12px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}

.logout-btn:hover {
  border-color: #f44;
  color: #f44;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #555;
  display: inline-block;
}

.status-dot.online {
  background: #4ade80;
  box-shadow: 0 0 6px #4ade80;
}

/* 欢迎区域 */
.welcome-section {
  padding: 32px 24px 16px;
}

.welcome-section h1 {
  margin: 0;
  font-size: 28px;
  color: #fff;
  font-weight: 300;
}

.sub-text {
  margin: 8px 0 0 0;
  color: #666;
  font-size: 14px;
}

/* 信息卡片 */
.info-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  padding: 0 24px;
  margin-bottom: 24px;
}

.info-card {
  background: #111;
  border: 1px solid #222;
  border-radius: 10px;
  padding: 20px;
  display: flex;
  align-items: center;
  gap: 16px;
  transition: border-color 0.3s;
}

.info-card:hover {
  border-color: #4ade80;
}

.info-card.blue { border-left: 3px solid #3b82f6; }
.info-card.green { border-left: 3px solid #4ade80; }
.info-card.purple { border-left: 3px solid #a855f7; }
.info-card.orange { border-left: 3px solid #f59e0b; }

.card-icon {
  font-size: 28px;
}

.card-value {
  margin: 0;
  font-size: 24px;
  font-weight: 700;
  color: #fff;
}

.card-label {
  margin: 4px 0 0 0;
  font-size: 12px;
  color: #666;
}

/* 功能入口 */
.action-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  padding: 0 24px;
  margin-bottom: 24px;
}

.action-card {
  background: #111;
  border: 1px solid #222;
  border-radius: 10px;
  padding: 24px;
  cursor: pointer;
  transition: all 0.3s;
  position: relative;
}

.action-card:hover {
  border-color: #4ade80;
  transform: translateY(-2px);
  box-shadow: 0 4px 20px rgba(74, 222, 128, 0.1);
}

.action-icon {
  font-size: 36px;
  margin-bottom: 12px;
}

.action-card h3 {
  margin: 0 0 8px 0;
  font-size: 18px;
  color: #fff;
}

.action-card p {
  margin: 0;
  font-size: 13px;
  color: #666;
  line-height: 1.5;
}

.action-status {
  margin-top: 16px;
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: #888;
}

/* 签到记录 */
.recent-section {
  margin: 0 24px;
  padding: 20px;
  background: #111;
  border: 1px solid #222;
  border-radius: 10px;
}

.recent-section h3 {
  margin: 0 0 16px 0;
  font-size: 16px;
  color: #fff;
}

.empty {
  color: #555;
  text-align: center;
  padding: 20px;
}

.log-list {
  max-height: 160px;
  overflow-y: auto;
}

.log-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 6px 0;
  font-size: 13px;
  color: #aaa;
  border-bottom: 1px solid #1a1a1a;
}

.log-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: #4ade80;
  flex-shrink: 0;
}
</style>