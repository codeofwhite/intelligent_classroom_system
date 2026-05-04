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

            <!-- 显示 班级 / 关联学生 -->
            <div class="extra" v-if="relationText">
              {{ relationText }}
            </div>
          </div>
        </div>

        <div class="menu-list">
          <router-link to="/" class="menu-item">🏠 首页</router-link>

          <template v-if="userRole === 'student'">
            <router-link to="/behavior" class="menu-item">📊 行为报告 & 学习画像</router-link>
            <router-link to="/medal" class="menu-item">🏅 荣誉勋章墙</router-link>
          </template>

          <template v-if="userRole === 'parent'">
            <router-link to="/report" class="menu-item">👶 孩子课堂报告</router-link>
            <router-link to="/suggest" class="menu-item">🏠 家校共育建议</router-link>
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
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import axios from 'axios'

const router = useRouter()
const route = useRoute()

const isLoggedIn = ref(false)
const userRole = ref('')
const userName = ref('')
const relationText = ref('') // 这里最终会显示：孩子：张三、李四

onMounted(async () => {
  const user = localStorage.getItem('currentUser')
  if (user) {
    try {
      const u = JSON.parse(user)
      isLoggedIn.value = true
      userRole.value = u.role
      userName.value = u.name

      // ==============================================
      // ✅ 家长：调用接口拿孩子列表（你就是缺这一步！）
      // ==============================================
      if (u.role === 'parent') {
        const res = await axios.post('http://localhost:5001/parent-children', {
          user_code: u.user_code
        })
        const children = res.data.children
        const names = children.map(c => c.student_name).join('、')
        relationText.value = '孩子：' + names
      }

      // 学生：显示班级（你原来的逻辑可以保留）
      else if (u.role === 'student') {
        relationText.value = '' // 你可以后续加班级逻辑
      }

    } catch (e) {
      console.error('加载关系失败', e)
    }
  }
})

// 登录成功后也调用接口加载孩子
const onLoginSuccess = async (userInfo) => {
  isLoggedIn.value = true
  userRole.value = userInfo.role
  userName.value = userInfo.name

  try {
    if (userInfo.role === 'parent') {
      const res = await axios.post('http://localhost:5001/parent-children', {
        user_code: userInfo.user_code
      })
      const children = res.data.children
      const names = children.map(c => c.student_name).join('、')
      relationText.value = '孩子：' + names
    }
  } catch (e) { }

  setTimeout(() => router.push('/'), 10)
}

const handleLogout = () => {
  localStorage.clear()
  isLoggedIn.value = false
  userRole.value = ''
  userName.value = ''
  relationText.value = ''
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

.full-page {
  width: 100%;
  min-height: 100vh;
}

.layout {
  display: flex;
  width: 100%;
  min-height: 100vh;
}

.sidebar {
  width: 240px;
  background: #fff;
  box-shadow: 0 0 12px rgba(0, 0, 0, 0.08);
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

.user-info .extra {
  margin-top: 4px;
  font-size: 12px;
  color: #999;
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

.content {
  flex: 1;
  margin-left: 240px;
  padding: 30px;
  width: calc(100% - 240px);
}
</style>