import { createRouter, createWebHistory } from 'vue-router'
import Index from '../views/Index.vue'
import FaceSign from '../views/FaceSign.vue'
import BehaviorMonitor from '../views/BehaviorMonitor.vue'
import Login from '../views/Login.vue'

const routes = [
  { path: '/', name: 'Index', component: Index },
  { path: '/face-sign', name: 'FaceSign', component: FaceSign },
  { path: '/behavior-monitor', name: 'BehaviorMonitor', component: BehaviorMonitor },
  { path: '/login', name: 'Login', component: Login }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router