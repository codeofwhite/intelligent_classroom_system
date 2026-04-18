<template>
  <div class="teacher-app">
    <aside v-if="route.path !== '/login'" class="sidebar">
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
      </nav>

      <div class="sidebar-footer">
        <button @click="handleLogout" class="logout-btn">
          退出登录
        </button>
      </div>
    </aside>

    <section class="main-container">
      <header v-if="route.path !== '/login'" class="top-header">
        <div class="breadcrumb">
          {{ currentClass }} /
          <span>{{ route.meta.title || '工作台' }}</span>
        </div>
        <div class="teacher-profile">
          <span class="notice">🔔</span>
          <span class="name">{{ teacherName }}</span>
          <div class="avatar">👨‍🏫</div>
        </div>
      </header>

      <main :class="{ content: route.path !== '/login', full: route.path === '/login' }">
        <router-view />
      </main>
    </section>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

const teacherName = ref('')
const currentClass = ref('三年级二班')

// 初始化：读取登录状态
onMounted(() => {
  const user = localStorage.getItem('userInfo')
  if (user) {
    const userObj = JSON.parse(user)
    teacherName.value = userObj.name || '老师'
  } else {
    if (route.path !== '/login') router.push('/login')
  }
})

// 登出
const handleLogout = () => {
  localStorage.removeItem('userInfo')
  router.push('/login')
}
</script>

<style scoped>
.teacher-app {
  display: flex;
  height: 100vh;
  background: #f5f7fa;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

.sidebar {
  width: 230px;
  background: #1e293b;
  color: #fff;
  display: flex;
  flex-direction: column;
  box-shadow: 2px 0 8px rgba(0, 0, 0, 0.1);
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

.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
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
}
.full {
  height: 100vh;
}
</style>