<template>
  <div class="login-container">
    <div class="login-box">
      <h2>🔑 欢迎回来</h2>
      <p>请选择身份并登录</p>

      <div class="role-tabs">
        <div 
          :class="['tab', loginForm.role === 'student' ? 'active' : '']" 
          @click="loginForm.role = 'student'"
        >我是学生</div>
        <div 
          :class="['tab', loginForm.role === 'parent' ? 'active' : '']" 
          @click="loginForm.role = 'parent'"
        >我是家长</div>
      </div>

      <div class="form-area">
        <input type="text" v-model="loginForm.username" placeholder="请输入姓名" />
        <input type="password" v-model="loginForm.password" placeholder="请输入密码" />
        <button class="login-btn" @click="handleLogin">立即登录</button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()

const loginForm = reactive({
  username: '',
  password: '',
  role: 'student' // 默认学生
})

const handleLogin = async () => {
  if (!loginForm.username || !loginForm.password) {
    alert('请填写完整信息')
    return
  }

  try {
    const response = await fetch('http://localhost:5001/login', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(loginForm)
    })

    const result = await response.json()

    if (result.status === 'success') {
      localStorage.setItem('currentUser', JSON.stringify(result.user))
      localStorage.setItem('currentRelations', JSON.stringify(result.relations || []))
      router.push('/')
    } else {
      alert(result.message)
    }
  } catch (error) {
    alert('后端服务未启动或网络异常')
  }
}
</script>

<style scoped>
.role-tabs {
  display: flex;
  background: #f0f2f5;
  border-radius: 12px;
  padding: 4px;
  margin-bottom: 24px;
}
.tab {
  flex: 1;
  padding: 10px;
  cursor: pointer;
  border-radius: 8px;
  transition: 0.3s;
  font-size: 14px;
}
.tab.active {
  background: white;
  color: #4a90e2;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.form-area {
  display: flex;
  flex-direction: column;
  gap: 16px;
}
input {
  padding: 14px;
  border: 1px solid #ddd;
  border-radius: 10px;
  outline: none;
}
input:focus {
  border-color: #4a90e2;
}
.login-btn {
  background: #4a90e2;
  color: white;
  padding: 14px;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  font-weight: bold;
}
/* ... 之前的 login-container 样式 ... */
</style>