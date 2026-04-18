<template>
  <div class="teacher-login-container">
    <div class="login-split">
      <div class="login-left">
        <div class="brand-info">
          <span class="logo-icon">🎓</span>
          <h1>智学堂·教师管理系统</h1>
          <p>赋能教学，科技让教育更简单</p>
        </div>
      </div>

      <div class="login-right">
        <div class="login-form-card">
          <h2>教师登录</h2>
          <p class="subtitle">欢迎回来，请登录您的账号</p>

          <form @submit.prevent="handleLogin">
            <div class="form-item">
              <label>工号/手机号</label>
              <input 
                v-model="loginForm.username" 
                type="text" 
                placeholder="请输入您的教职工号" 
                required
              />
            </div>

            <div class="form-item">
              <label>登录密码</label>
              <input 
                v-model="loginForm.password" 
                type="password" 
                placeholder="请输入密码" 
                required
              />
            </div>

            <button type="submit" class="login-submit-btn" :disabled="loading">
              {{ loading ? "登录中..." : "立即登录" }}
            </button>
          </form>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const loading = ref(false)

const loginForm = reactive({
  username: '',
  password: ''
})

const handleLogin = async () => {
  loading.value = true
  try {
    const res = await axios.post('http://localhost:5001/login', {
      username: loginForm.username,
      password: loginForm.password,
      role: 'teacher'
    })

    localStorage.setItem('userInfo', JSON.stringify(res.data.user))
    alert('登录成功！')
    router.push('/')
  } catch (err) {
    const msg = err.response?.data?.message || '登录失败'
    alert('失败：' + msg)
  } finally {
    loading.value = false
  }
}
</script>

<style scoped>
.teacher-login-container {
  width: 100vw;
  height: 100vh;
  background: #f0f2f5;
  display: flex;
  align-items: center;
  justify-content: center;
}

.login-split {
  display: flex;
  width: 1000px;
  height: 600px;
  background: white;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 20px 50px rgba(0,0,0,0.1);
}

.login-left {
  flex: 1.2;
  background: linear-gradient(135deg, #001529, #003a8c);
  color: white;
  padding: 60px;
  display: flex;
  align-items: center;
}

.logo-icon {
  font-size: 40px;
  margin-bottom: 20px;
}

.brand-info h1 {
  font-size: 28px;
  margin-bottom: 12px;
}

.brand-info p {
  opacity: 0.7;
}

.login-right {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 40px;
}

.login-form-card {
  width: 100%;
  max-width: 320px;
}

h2 {
  font-size: 24px;
  color: #333;
  margin-bottom: 8px;
}

.subtitle {
  color: #999;
  font-size: 14px;
  margin-bottom: 30px;
}

.form-item {
  margin-bottom: 20px;
}

.form-item label {
  display: block;
  font-size: 13px;
  color: #666;
  margin-bottom: 8px;
}

.form-item input {
  width: 100%;
  padding: 12px;
  border: 1px solid #d9d9d9;
  border-radius: 4px;
  outline: none;
}

.form-item input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24,144,255,0.2);
}

.login-submit-btn {
  width: 100%;
  padding: 12px;
  background-color: #1890ff;
  color: white;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
}

.login-submit-btn:hover {
  background-color: #40a9ff;
}
</style>