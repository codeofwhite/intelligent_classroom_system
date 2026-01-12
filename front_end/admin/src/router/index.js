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
    component: () => import('../views/LoginView.vue')
  },
  {
    path: '/videos',
    name: 'videos',
    component: () => import('../views/VideosView.vue'), 
    meta: { title: '课堂录像回放' }
  },
  {
    path: '/members',
    name: 'members',
    component: () => import('../views/MembersView.vue'),
    meta: { title: '班级成员管理' }
  },
  {
    path: '/reports',
    name: 'reports',
    component: () => import('../views/ReportsView.vue'),
    meta: { title: '学生行为报告' }
  }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router