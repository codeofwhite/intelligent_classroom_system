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
        <button @click="handleLogout" class="logout-btn">退出</button>
      </div>
    </nav>

    <main class="content">
      <router-view @login-success="onLoginSuccess" />
    </main>
  </div>
</template>

<style>
.nav-links a {
  text-decoration: none;
  color: #666;
  font-weight: 500;
  padding: 5px 10px;
  border-radius: 4px;
}

.router-link-active {
  color: #4a90e2 !important;
  background-color: #f0f7ff;
}

.content {
  padding: 20px;
  max-width: 1200px;
  margin: 0 auto;
}

body {
  margin: 0;
  font-family: 'PingFang SC', 'Microsoft YaHei', sans-serif;
  background-color: #f5f7fa;
}

#app {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* 导航栏样式 */
.main-navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 40px;
  height: 70px;
  background-color: #ffffff;
  box-shadow: 0 2px 10px rgba(0,0,0,0.05);
  position: sticky;
  top: 0;
  z-index: 100;
}

.nav-left {
  display: flex;
  align-items: center;
  gap: 40px;
}

.logo {
  font-size: 24px;
  font-weight: bold;
  color: #4a90e2;
  display: flex;
  align-items: center;
  gap: 8px;
}

.nav-links {
  display: flex;
  list-style: none;
  gap: 25px;
  margin: 0;
  padding: 0;
}

.nav-links li {
  cursor: pointer;
  color: #666;
  font-weight: 500;
  transition: color 0.3s;
}

.nav-links li:hover, .nav-links li.active {
  color: #4a90e2;
}

.nav-right {
  display: flex;
  align-items: center;
  gap: 20px;
}

/* 身份标签样式 */
.user-badge {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 5px 15px;
  border-radius: 20px;
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
  background: rgba(255,255,255,0.5);
  padding: 2px 6px;
  border-radius: 4px;
}

.logout-btn {
  background: none;
  border: 1px solid #ddd;
  padding: 5px 12px;
  border-radius: 6px;
  cursor: pointer;
  color: #888;
}

.content {
  flex: 1;
  width: 100%;
  max-width: 1200px; /* 居中显示，防止在大屏下太散 */
  margin: 0 auto;
}
</style>