import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'

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
    component: () => import('../views/LoginView.vue')
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router