<template>
  <div class="home-view">
    <header class="welcome-section">
      <div class="user-info">
        <!-- 头像 + 下拉菜单 -->
        <div class="avatar-container" @click="showMenu = !showMenu">
          <div class="avatar">🌟</div>
          <div v-if="showMenu" class="dropdown-menu">
            <button @click="handleLogout">退出登录</button>
          </div>
        </div>
        
        <!-- 欢迎文字 + 身份信息 -->
        <div class="text">
          <h2>早安，{{ user.username }}！</h2>
          <p v-if="relations.length > 0" class="relation-tag">
            身份：{{ user.role === 'parent' ? '家长' : '老师' }} 
            (关联学生: {{ relations.map(r => r.username).join(', ') }})
          </p>
          <p v-else class="welcome-desc">今天也是充满进步的一天，加油！</p>
        </div>
      </div>
    </header>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'

const router = useRouter()
const user = ref({})
const relations = ref([])
const showMenu = ref(false)

onMounted(() => {
  const savedUser = localStorage.getItem('currentUser')
  const savedRelations = localStorage.getItem('currentRelations')
  
  if (!savedUser) {
    router.push('/login')
    return
  }
  
  user.value = JSON.parse(savedUser)
  relations.value = JSON.parse(savedRelations || '[]')
})

// 登出
const handleLogout = () => {
  localStorage.removeItem('currentUser')
  localStorage.removeItem('currentRelations')
  router.push('/login')
}
</script>

<style scoped>
/* 整体页面 */
.home-view {
  width: 100%;
  min-height: 100vh;
  background-color: #f7f9fc;
}

/* 头部欢迎栏 */
.welcome-section {
  width: 100%;
  padding: 24px 20px;
  background: linear-gradient(135deg, #4e8cff, #64b5f6);
  color: white;
  box-sizing: border-box;
  box-shadow: 0 2px 10px rgba(78, 140, 255, 0.15);
}

/* 用户信息布局：横向排列 */
.user-info {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 16px;
}

/* 头像容器 */
.avatar-container {
  position: relative;
  cursor: pointer;
  flex-shrink: 0;
}

/* 头像样式 */
.avatar {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.25);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
  backdrop-filter: blur(10px);
  transition: transform 0.2s ease;
}

.avatar:hover {
  transform: scale(1.05);
}

/* 下拉菜单 */
.dropdown-menu {
  position: absolute;
  top: 55px;
  left: 0;
  background: #ffffff;
  border: 1px solid #f0f0f0;
  border-radius: 10px;
  box-shadow: 0 6px 20px rgba(0, 0, 0, 0.1);
  z-index: 100;
  overflow: hidden;
  min-width: 110px;
}

.dropdown-menu button {
  width: 100%;
  padding: 11px 18px;
  border: none;
  background: none;
  color: #ff4d4f;
  font-size: 14px;
  cursor: pointer;
  text-align: left;
  transition: background 0.2s;
}

.dropdown-menu button:hover {
  background-color: #fff5f5;
}

/* 文字区域 */
.text {
  flex: 1;
  line-height: 1.4;
}

.text h2 {
  margin: 0 0 6px 0;
  font-size: 20px;
  font-weight: 600;
}

/* 身份标签 */
.relation-tag {
  margin: 0;
  font-size: 13px;
  color: #3079ed;
  background: rgba(255, 255, 255, 0.85);
  padding: 4px 10px;
  border-radius: 12px;
  display: inline-block;
  backdrop-filter: blur(4px);
}

/* 温馨提示文字 */
.welcome-desc {
  margin: 0;
  font-size: 14px;
  opacity: 0.95;
}
</style>