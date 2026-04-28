<template>
  <div class="teacher-app">
    <!-- 登录页面：不显示侧边栏 + 顶部栏 -->
    <div v-if="route.path === '/login'" class="full-page">
      <router-view />
    </div>

    <!-- 已登录：显示侧边栏 + 布局 -->
    <div v-else class="layout">
      <aside class="sidebar">
        <div class="brand">
          <span class="logo">🎓</span>
          <span class="title">智慧课堂系统</span>
        </div>

        <nav class="menu">
          <router-link to="/" class="menu-item">
            <i class="icon">📊</i>
            <span class="label">工作台首页</span>
          </router-link>
          <router-link to="/videos" class="menu-item">
            <i class="icon">📹</i>
            <span class="label">课堂录像回放</span>
          </router-link>
          <router-link to="/members" class="menu-item">
            <i class="icon">👥</i>
            <span class="label">班级成员管理</span>
          </router-link>
          <router-link to="/reports" class="menu-item">
            <i class="icon">📝</i>
            <span class="label">学生行为报告</span>
          </router-link>
          <router-link to="/ai-chat" class="menu-item">
            <i class="icon">🤖</i>
            <span class="label">AI 课堂助手</span>
          </router-link>
        </nav>

        <div class="sidebar-footer">
          <button @click="handleLogout" class="logout-btn">
            退出登录
          </button>
        </div>
      </aside>

      <div class="main-container">
        <header class="top-header">
          <div class="breadcrumb">
            <span>{{ route.meta.title || '工作台' }}</span>
          </div>
          <div class="teacher-profile">
            <span class="notice">🔔</span>
            <span class="name">{{ teacherName }}</span>
            <div class="avatar">👨‍🏫</div>
          </div>
        </header>

        <main class="content">
          <router-view />
        </main>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

const teacherName = ref('')
const isLoggedIn = ref(false)

// 初始化登录状态
onMounted(() => {
  const user = localStorage.getItem('userInfo')
  if (user) {
    try {
      const u = JSON.parse(user)
      isLoggedIn.value = true
      teacherName.value = u.name || '老师'
    } catch (e) { }
  } else {
    if (route.path !== '/login') {
      router.push('/login')
    }
  }
})

// 登出
const handleLogout = () => {
  isLoggedIn.value = false
  teacherName.value = ''
  localStorage.removeItem('userInfo')
  router.push('/login')
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

.teacher-app {
  width: 100%;
  min-height: 100vh;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.full-page {
  width: 100%;
  height: 100vh;
}

.layout {
  display: flex;
  width: 100%;
  min-height: 100vh;
}

/* 侧边栏 */
.sidebar {
  width: 230px;
  background: #1e293b;
  color: #fff;
  display: flex;
  flex-direction: column;
  position: fixed;
  height: 100vh;
  left: 0;
  top: 0;
  z-index: 99;
}

.brand {
  padding: 22px 20px;
  display: flex;
  align-items: center;
  gap: 10px;
  background: #0f172a;
}

.logo {
  font-size: 22px;
}

.title {
  font-size: 17px;
  font-weight: 600;
}

.menu {
  flex: 1;
  padding: 12px 0;
}

.menu-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 14px 20px;
  margin: 4px 10px;
  border-radius: 8px;
  color: #cbd5e1;
  text-decoration: none;
  transition: all 0.25s ease;
}

.menu-item .icon {
  font-size: 18px;
  width: 20px;
  text-align: center;
}

.menu-item .label {
  font-size: 14px;
}

.menu-item:hover {
  background: #334155;
  color: #fff;
}

.router-link-active {
  background: #2563eb;
  color: #fff;
  font-weight: 500;
}

.sidebar-footer {
  padding: 16px;
  border-top: 1px solid #334155;
}

.logout-btn {
  width: 100%;
  padding: 10px;
  border-radius: 8px;
  background: #334155;
  color: #cbd5e1;
  border: none;
  cursor: pointer;
  font-size: 14px;
  transition: 0.2s;
}

.logout-btn:hover {
  background: #ef4444;
  color: #fff;
}

/* 右侧主体 */
.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  margin-left: 230px;
  width: calc(100% - 230px);
}

.top-header {
  height: 60px;
  background: #fff;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

.breadcrumb {
  font-size: 14px;
  color: #64748b;
}

.breadcrumb span {
  color: #2563eb;
  font-weight: 500;
}

.teacher-profile {
  display: flex;
  align-items: center;
  gap: 14px;
  font-size: 14px;
}

.avatar {
  width: 32px;
  height: 32px;
  background: #e2e8f0;
  border-radius: 50%;
  display: grid;
  place-items: center;
  font-size: 16px;
}

.content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
  background: #f5f7fa;
}
</style>