<template>
  <div class="ai-chat-page">
    <!-- 左侧：历史对话 -->
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
        </div>
      </div>
    </div>

    <!-- 右侧：聊天区域 -->
    <div class="chat-container">
      <div class="chat-box" ref="chatBox">
        <div v-for="(msg, idx) in messages" :key="idx" class="msg" :class="msg.role">
          <!-- 🔥 宽松匹配图片链接 -->
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
  </div>
</template>

<script setup>
import { ref, onMounted, nextTick } from 'vue'

const sessionList = ref([])
const currentSessionId = ref(null)
const messages = ref([])
const inputText = ref('')
const loading = ref(false)
const chatBox = ref(null)

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

// 🔥 修复：判断是否包含图片链接（更宽松）
function isImageMsg(content) {
  return content && /https?:\/\/.*\.(jpg|jpeg|png)/i.test(content)
}

// 🔥 修复：提取 URL（正则更通用）
function getImageUrl(content) {
  const match = content.match(/https?:\/\/[^\s]+/i)
  return match ? match[0] : ''
}

// 加载会话列表（从 MySQL）
async function loadSessionList() {
  const res = await fetch('http://localhost:5002/api/chat/sessions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ teacher_code: teacherCode })
  })
  const data = await res.json()
  sessionList.value = data.sessions || []
}

// 加载单条会话消息
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

// 新建对话
async function newChat() {
  const id = 'sess_' + Date.now()
  currentSessionId.value = id
  messages.value = [{ role: 'ai', content: '你好！我是课堂分析AI助手～' }]
  await loadSessionList()
}

// 切换会话
async function switchSession(sessionId) {
  currentSessionId.value = sessionId
  messages.value = await loadSessionMessages(sessionId)
  await nextTick()
  chatBox.value.scrollTop = chatBox.value.scrollHeight
}

// 发送消息
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
    await loadSessionList()
  } catch (err) {
    messages.value.push({ role: 'ai', content: '请求失败，请稍后重试' })
  }

  loading.value = false
  await nextTick()
  chatBox.value.scrollTop = chatBox.value.scrollHeight
}

// 时间格式化
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

.chat-sidebar {
  width: 240px;
  background: #fff;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
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

.chat-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
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

/* 图片样式 */
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
</style>