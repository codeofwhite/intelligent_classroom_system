<template>
  <div class="ai-chat-page">
    <div class="chat-container">
      <div class="chat-box" ref="chatBox">
        <div v-for="(msg, idx) in messages" :key="idx" class="msg" :class="msg.role">
          <div class="bubble">{{ msg.content }}</div>
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

const messages = ref([
  { role: 'ai', content: '你好！我是课堂分析AI助手，我可以查询你的历史课堂、专注度、分心统计等数据～' }
])

const inputText = ref('')
const loading = ref(false)
const chatBox = ref(null)

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
    const user = JSON.parse(localStorage.getItem('userInfo') || '{}')
    const teacherCode = user.teacher_code || 'T2025001'

    const res = await fetch('http://localhost:5002/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: q,
        teacher_code: teacherCode
      })
    })

    const data = await res.json()
    messages.value.push({ role: 'ai', content: data.answer })
  } catch (err) {
    messages.value.push({ role: 'ai', content: '请求失败，请稍后重试' })
  }

  loading.value = false
  await nextTick()
  chatBox.value.scrollTop = chatBox.value.scrollHeight
}
</script>

<style scoped>
.ai-chat-page {
  width: 100%;
  height: 100%;
  display: flex;
  justify-content: center;
  align-items: center;
}

.chat-container {
  width: 100%;
  max-width: 750px;
  height: 720px;
  background: #fff;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.06);
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