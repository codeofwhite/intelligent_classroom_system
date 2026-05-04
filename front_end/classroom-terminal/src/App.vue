<template>
  <div id="app">
    <nav v-if="route.path !== '/login'">
      <router-link to="/">🎮 控制台</router-link>
      <router-link to="/face-sign">👤 人脸识别签到</router-link>
      <router-link to="/behavior-monitor">📊 课堂实时监测</router-link>
      <button class="logout-btn" @click="logout">🚪 退出登录</button>
    </nav>
    <router-view />
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'

const router = useRouter()
const route = useRoute()

// 检查登录状态
onMounted(() => {
  const userInfo = localStorage.getItem('terminalUser')
  if (!userInfo && route.path !== '/login') {
    router.push('/login')
  }
})

// 退出登录
const logout = () => {
  localStorage.removeItem('terminalUser')
  router.push('/login')
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
#app {
  min-height: 100vh;
  background: #000;
  color: #0f0;
  font-family: "Courier New", monospace;
}
nav {
  padding: 1rem;
  background: #111;
  display: flex;
  gap: 20px;
  border-bottom: 1px solid #0f0;
}
nav a {
  color: #0f0;
  text-decoration: none;
  font-size: 18px;
}
nav a:hover {
  color: #fff;
}
.logout-btn {
  margin-left: auto;
  background: #222;
  color: #0f0;
  border: 1px solid #0f0;
  padding: 4px 10px;
  cursor: pointer;
}
</style>