<script setup>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

const isLoggedIn = ref(false) 
const userRole = ref('student')
const userName = ref('同学')

const onLoginSuccess = (role) => {
  isLoggedIn.value = true
  userRole.value = role
  userName.value = role === 'student' ? '小明同学' : '小明家长'
  router.push('/') 
}

const handleLogout = () => {
  isLoggedIn.value = false
  router.push('/login')
}
</script>

<template>
  <div id="app">
    <nav v-if="isLoggedIn && route.path !== '/login'" class="main-navbar">
      <div class="nav-left">
        <div class="logo">🚀 <span>智学堂</span></div>
        <ul class="nav-links">
          <li><router-link to="/">首页</router-link></li>
          <li><router-link to="/courses">我的课程</router-link></li>
          <li v-if="userRole === 'parent'">
            <router-link to="/child-stats">孩子进度</router-link>
          </li>
        </ul>
      </div>
      <div class="nav-right">
        <div class="user-badge" :class="userRole">
          <span class="role-tag">{{ userRole === 'student' ? '学生' : '家长' }}</span>
          <span>{{ userName }}</span>
        </div>
        <button @click="handleLogout" class="logout-btn">退出登录</button>
      </div>
    </nav>

    <main class="content">
      <router-view @login-success="onLoginSuccess" />
    </main>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background-color: #f7fcff;
  color: #333;
  line-height: 1.5;
}

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* 导航栏 */
.main-navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 30px;
  height: 65px;
  background: #ffffff;
  box-shadow: 0 2px 12px rgba(74, 144, 226, 0.06);
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 36px;
}

.logo {
  font-size: 22px;
  font-weight: bold;
  color: #4a90e2;
  display: flex;
  align-items: center;
  gap: 8px;
}

.nav-links {
  display: flex;
  list-style: none;
  gap: 24px;
}

.nav-links a {
  text-decoration: none;
  color: #555;
  font-weight: 500;
  padding: 8px 12px;
  border-radius: 8px;
  transition: all 0.25s ease;
}

.nav-links a:hover {
  color: #4a90e2;
  background: #f0f7ff;
}

.router-link-active {
  color: #4a90e2 !important;
  background-color: #e8f3ff !important;
}

.nav-right {
  display: flex;
  align-items: center;
  gap: 16px;
}

/* 身份卡片 */
.user-badge {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 6px 14px;
  border-radius: 30px;
  font-size: 14px;
}

.user-badge.student {
  background-color: #e1f5fe;
  color: #0288d1;
}

.user-badge.parent {
  background-color: #f3e5f5;
  color: #7b1fa2;
}

.role-tag {
  font-size: 12px;
  background: rgba(255, 255, 255, 0.6);
  padding: 2px 6px;
  border-radius: 4px;
}

/* 退出按钮 */
.logout-btn {
  border: none;
  padding: 8px 14px;
  border-radius: 10px;
  cursor: pointer;
  font-size: 13px;
  background: #ff6b6b;
  color: #fff;
  transition: 0.2s;
}
.logout-btn:hover {
  background: #ff5252;
}

/* 内容区域 */
.content {
  flex: 1;
  width: 100%;
  max-width: 1100px;
  margin: 0 auto;
  padding: 24px 20px;
}
</style>