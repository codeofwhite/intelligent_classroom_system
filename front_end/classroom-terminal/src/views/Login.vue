<template>
  <div class="login-page">
    <div class="login-box">
      <h2>> 教师终端登录</h2>

      <div class="item">
        <label>账号：</label>
        <input v-model="username" type="text" placeholder="输入教师账号" />
      </div>

      <div class="item">
        <label>密码：</label>
        <input v-model="password" type="password" placeholder="输入密码" />
      </div>

      <button class="login-btn" @click="login">
        {{ loading ? '登录中...' : '▶️ 登录系统' }}
      </button>

      <p class="tip">{{ msg }}</p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const username = ref('')
const password = ref('')
const loading = ref(false)
const msg = ref('')

const login = async () => {
  if (!username.value || !password.value) {
    msg.value = '请输入账号密码'
    return
  }

  loading.value = true
  msg.value = ''

  try {
    const res = await axios.post('http://localhost:5001/login', {
      username: username.value,
      password: password.value,
      role: 'teacher'
    })

    // 保存登录状态
    localStorage.setItem('terminalUser', JSON.stringify(res.data.user))
    router.push('/')
  } catch (err) {
    msg.value = err.response?.data?.message || '登录失败'
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.login-page {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #000;
  color: #0f0;
}
.login-box {
  border: 1px solid #0f0;
  padding: 30px;
  width: 380px;
  background: #111;
}
.item {
  margin: 16px 0;
}
input {
  width: 100%;
  padding: 8px;
  background: #222;
  color: #0f0;
  border: 1px solid #0f0;
  margin-top: 6px;
}
.login-btn {
  width: 100%;
  padding: 10px;
  background: #0f0;
  color: #000;
  border: none;
  cursor: pointer;
  font-weight: bold;
}
.tip {
  margin-top: 10px;
  color: #ff4444;
}
</style>