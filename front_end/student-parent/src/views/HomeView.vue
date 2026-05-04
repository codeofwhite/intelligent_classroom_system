<template>
  <div class="home-view">
    <!-- 顶部欢迎栏 -->
    <header class="welcome-section">
      <div class="user-info">
        <div class="avatar-container" @click="showMenu = !showMenu">
          <div class="avatar">🌟</div>
          <div v-if="showMenu" class="dropdown-menu">
            <button @click="handleLogout">退出登录</button>
          </div>
        </div>

        <div class="text">
          <h2>早安，{{ user.name }}！</h2>
          <p v-if="user.role === 'student'" class="relation-tag">
            身份：学生 | 班级：{{ studentInfo.class_name || '加载中...' }}
          </p>
          <p v-else-if="user.role === 'parent'" class="relation-tag">
            身份：家长（关联学生：{{ children.map(c => c.student_name).join(', ') }}）
          </p>
        </div>
      </div>
    </header>

    <!-- 自动加载对应首页 -->
    <div class="page-content">
      <StudentHome v-if="user.role === 'student'" />
      <ParentHome v-else-if="user.role === 'parent'" />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import axios from 'axios'
import StudentHome from '../components/StudentHome.vue'
import ParentHome from '../components/ParentHome.vue'

const router = useRouter()
const user = ref({})
const children = ref([])
const studentInfo = ref({ class_name: '' })
const showMenu = ref(false)

onMounted(async () => {
  const savedUser = localStorage.getItem('currentUser')
  if (!savedUser) return router.push('/login')

  user.value = JSON.parse(savedUser)

  // ==========================================
  // 家长：获取绑定的孩子
  // ==========================================
  if (user.value.role === 'parent') {
    try {
      const res = await axios.post('http://localhost:5001/parent-children', {
        user_code: user.value.user_code
      })
      children.value = res.data.children || []
    } catch (err) {
      console.error('获取孩子失败', err)
    }
  }

  // ==========================================
  // 学生：获取班级信息（真实接口）
  // ==========================================
  if (user.value.role === 'student') {
    try {
      const res = await axios.get('http://localhost:5002/api/student/info', {
        params: { student_code: user.value.student_code }
      })
      studentInfo.value = res.data
    } catch (err) {
      console.error('获取学生信息失败', err)
    }
  }
})

const handleLogout = () => {
  localStorage.clear()
  router.push('/login')
}
</script>

<style scoped>
.welcome-section {
  width: 100%;
  padding: 24px 20px;
  background: linear-gradient(135deg, #4e8cff, #64b5f6);
  color: white;
  box-shadow: 0 2px 10px rgba(78, 140, 255, 0.15);
}
.user-info {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  align-items: center;
  gap: 16px;
}
.avatar-container {
  position: relative;
  cursor: pointer;
}
.avatar {
  width: 50px;
  height: 50px;
  border-radius: 50%;
  background: rgba(255, 255, 255, 0.25);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 22px;
}
.dropdown-menu {
  position: absolute;
  top: 55px;
  background: #fff;
  border-radius: 10px;
  box-shadow: 0 6px 20px rgba(0,0,0,0.1);
  padding: 5px;
  z-index: 99;
}
.dropdown-menu button {
  padding: 10px 15px;
  border: none;
  background: none;
  color: #ff4d4f;
  cursor: pointer;
}
.text h2 {
  margin: 0 0 6px;
  font-size: 20px;
}
.relation-tag {
  font-size: 13px;
  background: rgba(255,255,255,0.9);
  padding: 4px 10px;
  border-radius: 12px;
  color: #4e8cff;
  display: inline-block;
}

.page-content {
  max-width: 1200px;
  margin: 30px auto;
  padding: 0 20px;
}
</style>