<template>
  <div class="teacher-login-container">
    <div class="login-split">
      <div class="login-left">
        <div class="brand-info">
          <span class="logo-icon">🎓</span>
          <h1>智学堂·教师管理系统</h1>
          <p>赋能教学，科技让教育更简单</p>
        </div>
        <div class="illustration">
          
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
                v-model="loginForm.id" 
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

            <div class="form-options">
              <label><input type="checkbox"> 记住我</label>
              <a href="#">忘记密码？</a>
            </div>

            <button type="submit" class="login-submit-btn">
              立即登录
            </button>
          </form>

          <div class="other-login">
            <span>其他登录方式</span>
            <div class="icons">
              <span class="wechat">微信扫码</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { reactive } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const loginForm = reactive({
  id: '',
  password: ''
})

const handleLogin = () => {
  // 暂时不连数据库，点击即通过
  console.log('教师登录信息：', loginForm)
  // 模拟登录成功后的跳转
  router.push('/')
}
</script>

<style scoped>
.teacher-login-container {
  height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #f0f2f5;
  margin: 0;
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

/* 左侧装饰区 */
.login-left {
  flex: 1.2;
  background: linear-gradient(135deg, #001529, #003a8c);
  color: white;
  padding: 60px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
}

.logo-icon {
  font-size: 40px;
  margin-bottom: 20px;
  display: block;
}

.brand-info h1 {
  font-size: 28px;
  margin-bottom: 12px;
}

.brand-info p {
  opacity: 0.7;
  letter-spacing: 1px;
}

/* 右侧表单区 */
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
  transition: all 0.3s;
}

.form-item input:focus {
  border-color: #1890ff;
  box-shadow: 0 0 0 2px rgba(24,144,255,0.2);
}

.form-options {
  display: flex;
  justify-content: space-between;
  font-size: 13px;
  margin-bottom: 25px;
}

.form-options a {
  color: #1890ff;
  text-decoration: none;
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
  transition: background 0.3s;
}

.login-submit-btn:hover {
  background-color: #40a9ff;
}

.other-login {
  margin-top: 40px;
  text-align: center;
  font-size: 12px;
  color: #999;
}

.wechat {
  display: inline-block;
  margin-top: 10px;
  color: #52c41a;
  cursor: pointer;
}
</style>