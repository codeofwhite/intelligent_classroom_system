<template>
  <div class="login-page">
    <div class="login-card">
      <div class="login-header">
        <div class="logo-icon">🚀</div>
        <h2>智学堂</h2>
        <p>智能课堂行为分析系统</p>
      </div>

      <div class="login-form">
        <p class="sub-title">请选择身份登录</p>

        <!-- 身份选择 -->
        <div class="role-tabs">
          <div 
            class="tab" 
            :class="loginForm.role === 'student' ? 'active' : ''"
            @click="loginForm.role = 'student'"
          >
            👨‍🎓 我是学生
          </div>
          <div 
            class="tab" 
            :class="loginForm.role === 'parent' ? 'active' : ''"
            @click="loginForm.role = 'parent'"
          >
            👨‍👩‍👧 我是家长
          </div>
        </div>

        <!-- 输入框 -->
        <div class="input-group">
          <input 
            type="text" 
            v-model="loginForm.username" 
            placeholder="请输入姓名"
          />
          <input 
            type="password" 
            v-model="loginForm.password" 
            placeholder="请输入密码"
          />
        </div>

        <!-- 登录按钮 -->
        <button class="login-btn" @click="handleLogin">
          安全登录
        </button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const emit = defineEmits(['login-success'])

const loginForm = reactive({
  username: '',
  password: '',
  role: 'student'
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
      emit('login-success', result.user)
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
/* 全屏背景 */
.login-page {
  width: 100vw;
  height: 100vh;
  background: linear-gradient(135deg, #f0f9ff 0%, #e6f7ff 100%);
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 登录卡片 */
.login-card {
  width: 420px;
  background: #ffffff;
  border-radius: 20px;
  box-shadow: 0 10px 40px rgba(0, 86, 179, 0.1);
  padding: 50px 40px;
  box-sizing: border-box;
  text-align: center;
  animation: fadeIn 0.4s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}

/* 头部 LOGO */
.login-header {
  margin-bottom: 32px;
}
.logo-icon {
  font-size: 40px;
  margin-bottom: 8px;
}
.login-header h2 {
  font-size: 26px;
  font-weight: 700;
  color: #1f2937;
  margin: 0;
}
.login-header p {
  font-size: 14px;
  color: #6b7280;
  margin: 6px 0 0;
}

/* 副标题 */
.sub-title {
  font-size: 15px;
  color: #4b5563;
  margin-bottom: 18px;
}

/* 身份选项卡 */
.role-tabs {
  display: flex;
  background: #f3f4f6;
  border-radius: 12px;
  padding: 4px;
  gap: 4px;
  margin-bottom: 24px;
}
.tab {
  flex: 1;
  padding: 12px 10px;
  border-radius: 10px;
  cursor: pointer;
  font-size: 15px;
  color: #6b7280;
  transition: all 0.25s ease;
}
.tab.active {
  background: #4e8cff;
  color: #fff;
  box-shadow: 0 2px 8px rgba(78, 140, 255, 0.25);
}

/* 输入框组 */
.input-group {
  display: flex;
  flex-direction: column;
  gap: 14px;
  margin-bottom: 24px;
}
.input-group input {
  padding: 15px 16px;
  border: 1px solid #e5e7eb;
  border-radius: 12px;
  font-size: 15px;
  outline: none;
  transition: border 0.2s;
}
.input-group input:focus {
  border-color: #4e8cff;
  box-shadow: 0 0 0 3px rgba(78, 140, 255, 0.1);
}

/* 登录按钮 */
.login-btn {
  width: 100%;
  padding: 16px;
  background: linear-gradient(135deg, #4e8cff, #3b82f6);
  color: white;
  font-size: 16px;
  font-weight: 600;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  transition: all 0.25s ease;
  box-shadow: 0 4px 12px rgba(78, 140, 255, 0.2);
}
.login-btn:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(78, 140, 255, 0.25);
}
.login-btn:active {
  transform: translateY(0);
}
</style>