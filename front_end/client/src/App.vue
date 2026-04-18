<template>
  <div id="app">
    <!-- 登录页面时：完全隐藏侧边栏，全屏显示 -->
    <div v-if="route.path === '/login'" class="full-page">
      <router-view @login-success="onLoginSuccess" />
    </div>

    <!-- 非登录页面：显示侧边栏 + 内容 -->
    <div v-else class="layout">
      <!-- 侧边栏 -->
      <aside class="sidebar">
        <div class="sidebar-header">
          <div class="logo">🚀 智学堂</div>
          <div class="user-info">
            <div class="role">{{ userRole === 'student' ? '学生' : '家长' }}</div>
            <div class="name">{{ userName }}</div>
          </div>
        </div>

        <div class="menu-list">
          <router-link to="/" class="menu-item">🏠 首页</router-link>

          <template v-if="userRole === 'student'">
            <router-link to="/behavior" class="menu-item">📊 行为报告 & 学习画像</router-link>
            <router-link to="/medal" class="menu-item">🏅 荣誉勋章墙</router-link>
          </template>

          <template v-if="userRole === 'parent'">
            <router-link to="/report" class="menu-item">📈 学情报告单</router-link>
            <router-link to="/suggest" class="menu-item">💡 家校共育建议</router-link>
            <router-link to="/warning" class="menu-item">⚠️ 异常行为摘要</router-link>
          </template>
        </div>

        <div class="sidebar-footer">
          <button @click="handleLogout" class="logout-btn">🚪 退出登录</button>
        </div>
      </aside>

      <!-- 内容区 -->
      <main class="content">
        <router-view @login-success="onLoginSuccess" />
      </main>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, watch } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

// 登录状态
const isLoggedIn = ref(false)
const userRole = ref('')
const userName = ref('')

// 页面刷新时自动恢复登录状态
onMounted(() => {
  const user = localStorage.getItem('currentUser')
  if (user) {
    try {
      const u = JSON.parse(user)
      isLoggedIn.value = true
      userRole.value = u.role
      userName.value = u.username
    } catch (e) {}
  }
})

// =============== 关键修复：登录成功后立刻更新，不延迟 ===============
const onLoginSuccess = (userInfo) => {
  // 直接赋值，强制响应式更新
  isLoggedIn.value = true
  userRole.value = userInfo.role
  userName.value = userInfo.username
  
  // 等下一帧再跳转，确保侧边栏渲染出来
  setTimeout(() => {
    router.push('/')
  }, 10)
}

// 登出
const handleLogout = () => {
  isLoggedIn.value = false
  userRole.value = ''
  userName.value = ''
  localStorage.clear()
  router.push('/login')
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
body {
  font-family: 'Microsoft YaHei', sans-serif;
  background: #f7f9fc;
  color: #333;
}
#app {
  width: 100%;
  min-height: 100vh;
}


/* 登录页面全屏 */
.full-page {
  width: 100%;
  min-height: 100vh;
}

/* 主布局 */
.layout {
  display: flex;
  width: 100%;
  min-height: 100vh;
}

/* 侧边栏 */
.sidebar {
  width: 240px;
  background: #fff;
  box-shadow: 0 0 12px rgba(0,0,0,0.08);
  display: flex;
  flex-direction: column;
  position: fixed;
  height: 100vh;
  left: 0;
  top: 0;
  z-index: 99;
}
.sidebar-header {
  padding: 24px 20px;
  border-bottom: 1px solid #f0f0f0;
}
.logo {
  font-size: 20px;
  font-weight: bold;
  color: #4e8cff;
  margin-bottom: 16px;
}
.user-info {
  font-size: 13px;
}
.user-info .role {
  color: #4e8cff;
  font-weight: bold;
}
.user-info .name {
  color: #666;
  margin-top: 4px;
}

.menu-list {
  flex: 1;
  padding: 20px 0;
}
.menu-item {
  display: block;
  padding: 12px 24px;
  text-decoration: none;
  color: #555;
  font-size: 14px;
  transition: 0.2s;
}
.menu-item:hover {
  background: #f0f7ff;
  color: #4e8cff;
}
.menu-item.router-link-active {
  background: #e8f3ff;
  color: #4e8cff;
  font-weight: 500;
}

.sidebar-footer {
  padding: 16px 20px;
  border-top: 1px solid #f0f0f0;
}
.logout-btn {
  width: 100%;
  padding: 10px;
  border: none;
  border-radius: 8px;
  background: #ff6b6b;
  color: #fff;
  cursor: pointer;
}
.logout-btn:hover {
  background: #ff5252;
}

/* 内容区 */
.content {
  flex: 1;
  margin-left: 240px;
  padding: 30px;
  width: calc(100% - 240px);
}
</style>