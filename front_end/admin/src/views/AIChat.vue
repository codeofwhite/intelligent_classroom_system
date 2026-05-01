<template>
  <div class="ai-chat-page">
    <!-- 左栏：历史对话 -->
    <div class="chat-sidebar">
      <div class="sidebar-header">
        <button class="new-chat-btn" @click="newChat">➕ 新对话</button>
      </div>
      <div class="session-list">
        <div
          v-for="s in sessionList"
          :key="s.session_id"
          class="session-item"
          :class="{ active: s.session_id === currentSessionId }"
          @click="switchSession(s.session_id)"
        >
          <div class="title">{{ s.title }}</div>
          <div class="time">{{ formatTime(s.update_time) }}</div>
          <div class="del-btn" @click.stop="deleteSession(s)">×</div>
        </div>
      </div>
    </div>

    <!-- 中间：聊天区域 -->
    <div class="chat-container">
      <div class="chat-box" ref="chatBox">
        <div v-for="(msg, idx) in messages" :key="idx" class="msg" :class="msg.role">
          <div v-if="isImageMsg(msg.content)" class="image-wrapper">
            <img :src="getImageUrl(msg.content)" alt="课堂关键帧" />
          </div>
          <div v-else class="bubble">{{ msg.content }}</div>
        </div>
        <div v-if="loading" class="loading">AI 思考中...</div>
      </div>

      <div class="chat-input-bar">
        <input
          v-model="inputText"
          @keyup.enter="sendMessage"
          placeholder="请输入问题，例如：我的历史课堂怎么样？"
        />
        <button @click="sendMessage" :disabled="loading">发送</button>
      </div>
    </div>

    <!-- 右栏：固定AI思维链面板 永久展示 不遮挡 -->
    <div class="thinking-panel">
      <div class="thinking-header">
        <h4>🧠 AI 思维链</h4>
      </div>
      <div class="thinking-body">
        <div class="section">
          <label>🎯 识别意图</label>
          <div class="val">{{ thinking.intent || '暂无' }}</div>
        </div>

        <div class="section" v-if="thinking.tools.length > 0">
          <label>⚙️ 工具调用流程</label>
          <div class="tool-item" v-for="(t, i) in thinking.tools" :key="i">
            <div class="name">{{ t }}</div>
            <div class="args">参数：{{ thinking.args[i] || '无' }}</div>
            <div class="result">返回：{{ thinking.results[i] || '无' }}</div>
          </div>
        </div>

        <div class="empty-tip" v-else>
          发送问题后自动展示智能体推理过程
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'
import axios from 'axios'

const sessionList = ref([])
const currentSessionId = ref(null)
const messages = ref([])
const inputText = ref('')
const loading = ref(false)
const chatBox = ref(null)

// 思维过程数据
const thinking = ref({
  intent: '',
  tools: [],
  args: [],
  results: []
})

let teacherCode = ''

onMounted(async () => {
  const user = JSON.parse(localStorage.getItem('userInfo') || '{}')
  teacherCode = user.teacher_code || 'T2025001'
  await loadSessionList()

  if (sessionList.value.length > 0) {
    switchSession(sessionList.value[0].session_id)
  } else {
    newChat()
  }
})

function isImageMsg(content) {
  return content && /https?:\/\/.*\.(jpg|jpeg|png)/i.test(content)
}
function getImageUrl(content) {
  const match = content.match(/https?:\/\/[^\s]+/i)
  return match ? match[0] : ''
}

async function loadSessionList() {
  const res = await fetch('http://localhost:5002/api/chat/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ teacher_code: teacherCode })
  })
  const data = await res.json()
  sessionList.value = data.sessions || []
}

async function loadSessionMessages(sessionId) {
  const res = await fetch('http://localhost:5002/api/chat/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      teacher_code: teacherCode,
      session_id: sessionId
    })
  })
  const data = await res.json()
  return data.messages || [{ role: 'ai', content: '你好！我是课堂分析AI助手～' }]
}

async function newChat() {
  const id = 'sess_' + Date.now()
  currentSessionId.value = id
  messages.value = [{ role: 'ai', content: '你好！我是课堂分析AI助手～' }]
  // 清空思维面板
  thinking.value = { intent: '', tools: [], args: [], results: [] }
  await loadSessionList()
}

async function switchSession(sessionId) {
  currentSessionId.value = sessionId
  messages.value = await loadSessionMessages(sessionId)
  // 切换会话清空思维
  thinking.value = { intent: '', tools: [], args: [], results: [] }
  await nextTick()
  chatBox.value.scrollTop = chatBox.value.scrollHeight
}

const sendMessage = async () => {
  const q = inputText.value.trim()
  if (!q || loading.value) return

  messages.value.push({ role: 'user', content: q })
  inputText.value = ''
  loading.value = true
  await nextTick()
  chatBox.value.scrollTop = chatBox.value.scrollHeight

  try {
    const res = await fetch('http://localhost:5002/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        teacher_code: teacherCode,
        session_id: currentSessionId.value
      })
    })

    const data = await res.json()
    messages.value.push({ role: 'ai', content: data.answer })

    // 赋值思维过程 自动刷新右侧面板
    thinking.value = data.thinking_process || {
      intent: '',
      tools: [],
      args: [],
      results: []
    }

    await loadSessionList()
  } catch (err) {
    messages.value.push({ role: 'ai', content: '请求失败，请稍后重试' })
  }

  loading.value = false
  await nextTick()
  chatBox.value.scrollTop = chatBox.value.scrollHeight
}

async function deleteSession(session) {
  if (!confirm('确定要删除这条对话记录吗？')) return
  try {
    await axios.post('http://localhost:5002/api/chat/delete_session', {
      teacher_code: teacherCode,
      session_id: session.session_id
    })
    await loadSessionList()
    if (currentSessionId.value === session.session_id) {
      messages.value = [{ role: 'ai', content: '你好！我是课堂分析AI助手～' }]
      thinking.value = { intent: '', tools: [], args: [], results: [] }
      currentSessionId.value = null
    }
  } catch (e) {
    alert('删除失败')
  }
}

function formatTime(timeStr) {
  if (!timeStr) return ''
  const d = new Date(timeStr)
  return `${d.getHours().toString().padStart(2, '0')}:${d.getMinutes().toString().padStart(2, '0')}`
}
</script>

<style scoped>
.ai-chat-page {
  width: 100%;
  height: 100%;
  display: flex;
  background: #f5f7fa;
  border-radius: 12px;
  overflow: hidden;
}

/* 左栏固定宽度 */
.chat-sidebar {
  width: 240px;
  background: #fff;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
}

.sidebar-header {
  padding: 16px;
}

.new-chat-btn {
  width: 100%;
  padding: 10px;
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  font-size: 14px;
}

.session-list {
  flex: 1;
  overflow-y: auto;
  padding: 8px;
}

.session-item {
  position: relative;
  padding: 10px 12px;
  border-radius: 8px;
  cursor: pointer;
  margin-bottom: 6px;
}

.session-item.active {
  background: #eff6ff;
}

.session-item .title {
  font-size: 14px;
  color: #1e293b;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.session-item .time {
  font-size: 12px;
  color: #64748b;
  margin-top: 2px;
}

.del-btn {
  position: absolute;
  right: 8px;
  top: 50%;
  transform: translateY(-50%);
  width: 18px;
  height: 18px;
  line-height: 18px;
  text-align: center;
  font-size: 16px;
  color: #999;
  border-radius: 50%;
  cursor: pointer;
}
.del-btn:hover {
  background: #fef2f2;
  color: #ef4444;
}

/* 中间聊天 自适应占满剩余 */
.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid #e5e7eb;
}

.chat-box {
  flex: 1;
  padding: 20px;
  overflow-y: auto;
  background: #f9fafb;
}

.msg {
  display: flex;
  margin-bottom: 12px;
}

.msg.user {
  justify-content: flex-end;
}

.msg.ai {
  justify-content: flex-start;
}

.bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 12px;
  font-size: 14px;
  line-height: 1.4;
}

.image-wrapper {
  max-width: 300px;
}
.image-wrapper img {
  width: 100%;
  border-radius: 12px;
  box-shadow: 0 1px 5px rgba(0,0,0,0.1);
}

.ai .bubble {
  background: #e2e8f0;
  color: #1e293b;
  border-bottom-left-radius: 4px;
}

.user .bubble {
  background: #2563eb;
  color: #fff;
  border-bottom-right-radius: 4px;
}

.chat-input-bar {
  display: flex;
  padding: 14px;
  gap: 8px;
  border-top: 1px solid #e5e7eb;
}

.chat-input-bar input {
  flex: 1;
  padding: 12px 14px;
  border: 1px solid #d1d5db;
  border-radius: 8px;
  font-size: 14px;
  outline: none;
}

.chat-input-bar button {
  padding: 0 18px;
  background: #2563eb;
  color: #fff;
  border: none;
  border-radius: 8px;
  font-size: 14px;
  cursor: pointer;
}

.chat-input-bar button:disabled {
  background: #94a3b8;
  cursor: not-allowed;
}

.loading {
  font-size: 13px;
  color: #64748b;
  padding: 8px 0;
  text-align: center;
}

/* 右栏：固定思维面板 宽度固定 不悬浮不遮挡 */
.thinking-panel {
  width: 340px;
  flex-shrink: 0;
  height: 100%;
  background: #1e1e2e;
  color: #fff;
  padding: 16px;
  overflow-y: auto;
}

.thinking-header h4 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: #fff;
}

.section {
  margin-bottom: 14px;
}
.section label {
  font-size: 12px;
  color: #9ca3af;
  margin-bottom: 4px;
  display: block;
}
.val {
  background: #313141;
  padding: 8px 10px;
  border-radius: 6px;
  font-size: 13px;
  word-break: break-all;
}

.tool-item {
  background: #29293d;
  padding: 10px;
  border-radius: 8px;
  margin-bottom: 8px;
  font-size: 12px;
}
.tool-item .name {
  color: #38bdf8;
  font-weight: bold;
  margin-bottom: 4px;
}
.tool-item .args {
  color: #a5b4fc;
  margin-bottom: 4px;
}
.tool-item .result {
  color: #d1d5db;
  word-break: break-all;
}

.empty-tip {
  color: #6b7280;
  font-size: 13px;
  text-align: center;
  padding: 20px 0;
}
</style>