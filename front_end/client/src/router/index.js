import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import LoginView from '../views/LoginView.vue'

// 学生页面
import Behavior from '../views/student/Behavior.vue'
import Medal from '../views/student/Medal.vue'

// 家长页面
import Report from '../views/parent/Report.vue'
import Suggest from '../views/parent/Suggest.vue'

const routes = [
  {
    path: '/',
    name: 'home',
    component: HomeView
  },
  {
    path: '/login',
    name: 'login',
    // 路由懒加载：只有访问登录页时才加载该组件
    component: LoginView
  },

  // 学生
  { path: '/behavior', component: Behavior },
  { path: '/medal', component: Medal },

  // 家长
  { path: '/report', component: Report },
  { path: '/suggest', component: Suggest },
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router