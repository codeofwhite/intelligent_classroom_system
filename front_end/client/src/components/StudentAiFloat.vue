<template>
  <div class="float-btn" @click="showModal = true">
    🤖
  </div>

  <div class="ai-modal" v-if="showModal">
    <div class="modal-wrap" @click.stop>
      <div class="ai-header">
        <span>🤖 学习小助手</span>
        <div class="ctrl">
          <span @click="showModal = false">×</span>
        </div>
      </div>

      <!-- 聊天列表：按顺序一条一条渲染 -->
      <div class="chat-list" ref="chatRef">
        <div class="msg ai">
          <div class="bubble">你好～我是你的学习AI助手，有什么想问我的吗？</div>
        </div>

        <div class="msg" v-for="item in chatList" :key="item.id">
          <div class="bubble" :class="item.type">
            {{ item.content }}
          </div>
        </div>
      </div>

      <div class="input-bar">
        <input v-model="msg" placeholder="输入问题..." @keyup.enter="send" />
        <button @click="send">发送</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'
import axios from 'axios'

const showModal = ref(false)
const msg = ref('')
// 改成单个数组存整条对话：type user / ai
const chatList = ref([])
const chatRef = ref(null)

let student = null
try {
  student = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) { }

async function send() {
  if (!msg.value.trim()) return
  if (!student || !student.student_code) {
    alert('请先登录')
    return
  }

  // 1. 先把用户消息加入列表
  const text = msg.value.trim()
  chatList.value.push({
    id: Date.now(),
    type: 'user',
    content: text
  })
  msg.value = ''

  // 滚动到底部
  await nextTick()
  chatRef.value.scrollTop = chatRef.value.scrollHeight

  // 2. 请求后端AI
  try {
    const res = await axios.post('http://localhost:5002/api/student/ai', {
      question: text,
      student_code: student.student_code
    })
    // AI回复加入列表
    chatList.value.push({
      id: Date.now() + 1,
      type: 'ai',
      content: res.data.answer
    })
  } catch (e) {
    chatList.value.push({
      id: Date.now() + 1,
      type: 'ai',
      content: '服务异常，请稍后再试'
    })
  }

  // 再次滚动到底部
  await nextTick()
  chatRef.value.scrollTop = chatRef.value.scrollHeight
}
</script>

<style scoped>
.float-btn {
  position: fixed;
  right: 24px;
  bottom: 30px;
  width: 52px;
  height: 52px;
  background: #5b8def;
  color: white;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
  cursor: pointer;
  z-index: 9999;
  box-shadow: 0 4px 12px #0002;
}

.ai-modal {
  position: fixed;
  inset: 0;
  background: #0003;
  z-index: 9999;
  display: flex;
  align-items: flex-end;
  justify-content: flex-end;
  padding: 16px;
}

.modal-wrap {
  width: 340px;
  height: 560px;
  background: white;
  border-radius: 16px;
  box-shadow: 0 10px 30px #0003;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.ai-header {
  background: #5b8def;
  color: white;
  padding: 14px 16px;
  font-weight: bold;
  display: flex;
  justify-content: space-between;
}

.chat-list {
  flex: 1;
  padding: 12px;
  background: #f7f8fa;
  overflow-y: auto;
}

/* 每条消息容器 */
.msg {
  margin-bottom: 12px;
  max-width: 75%;
}

/* AI居左 */
.msg.ai {
  margin-right: auto;
}

/* 用户居右 */
.msg.user {
  margin-left: auto;
}

.bubble {
  padding: 10px 14px;
  border-radius: 14px;
  font-size: 14px;
  line-height: 1.6;
}

.bubble.ai {
  background: #eef4ff;
  color: #333;
}

.bubble.user {
  background: #5b8def;
  color: #fff;
}

.input-bar {
  display: flex;
  padding: 10px;
  gap: 8px;
  border-top: 1px solid #eee;
}

input {
  flex: 1;
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 20px;
  outline: none;
}

button {
  padding: 8px 14px;
  background: #5b8def;
  color: white;
  border: none;
  border-radius: 20px;
  cursor: pointer;
}
</style>