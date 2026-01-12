<script setup>
import { ref } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

const teacherName = ref('王老师')
const currentClass = ref('三年级二班')

// 退出登录逻辑
const handleLogout = () => {
  // 清除本地状态（以后连数据库时清理 token）
  router.push('/login')
}
</script>

<template>
  <div class="teacher-app">
    <aside v-if="route.path !== '/login'" class="sidebar">
      <div class="brand">
        <span class="icon">🎓</span>
        <span class="title">教师管理后台</span>
      </div>
      
      <nav class="menu">
        <router-link to="/" class="menu-item">
          <i class="icon">📊</i> 工作台首页
        </router-link>
        <router-link to="/videos" class="menu-item">
          <i class="icon">📹</i> 课堂录像回放
        </router-link>
        <router-link to="/members" class="menu-item">
          <i class="icon">👥</i> 班级成员管理
        </router-link>
        <router-link to="/reports" class="menu-item">
          <i class="icon">📝</i> 学生行为报告
        </router-link>
        <router-link to="/messages" class="menu-item">
          <i class="icon">💬</i> 家长沟通
        </router-link>
      </nav>
      
      <div class="sidebar-footer">
        <button @click="handleLogout" class="logout-btn">退出登录</button>
      </div>
    </aside>

    <section class="main-container">
      <header v-if="route.path !== '/login'" class="top-header">
        <div class="breadcrumb">
          {{ currentClass }} / 
          <span>{{ route.meta.title || '工作台' }}</span>
        </div>
        <div class="teacher-profile">
          <span class="notice-badge">🔔</span>
          <span class="name">{{ teacherName }}</span>
          <div class="avatar">👨‍🏫</div>
        </div>
      </header>

      <main :class="{ 'content-view': route.path !== '/login', 'full-page': route.path === '/login' }">
        <router-view />
      </main>
    </section>
  </div>
</template>

<style scoped>
.teacher-app {
  display: flex;
  height: 100vh;
  background-color: #f0f2f5;
}

/* 侧边栏 */
.sidebar {
  width: 240px;
  background-color: #001529;
  color: white;
  display: flex;
  flex-direction: column;
}

.brand {
  padding: 24px;
  font-size: 20px;
  font-weight: bold;
  background: #002140;
  display: flex;
  align-items: center;
  gap: 10px;
}

.menu {
  flex: 1;
  padding: 16px 0;
}

.menu-item {
  display: flex;
  align-items: center;
  padding: 14px 24px;
  color: #a6adb4;
  text-decoration: none;
  transition: all 0.3s;
}

.menu-item:hover, .router-link-active {
  color: white;
  background-color: #1890ff;
}

/* 右侧内容 */
.main-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.top-header {
  height: 64px;
  background: white;
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 0 24px;
  box-shadow: 0 1px 4px rgba(0,21,41,0.08);
}

.content-view {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
}

.sidebar-footer {
  padding: 20px;
  border-top: 1px solid #002140;
}

.logout-btn {
  width: 100%;
  padding: 8px;
  background: transparent;
  border: 1px solid #444;
  color: #a6adb4;
  cursor: pointer;
  border-radius: 4px;
}

.logout-btn:hover {
  color: white;
  border-color: #ff4d4f;
  background: #ff4d4f;
}

.full-page {
  width: 100%;
  height: 100vh;
}

.breadcrumb span {
  color: #1890ff;
  margin-left: 8px;
}
</style>