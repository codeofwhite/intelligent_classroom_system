import { createRouter, createWebHistory } from 'vue-router'
import Index from '../views/Index.vue'
import FaceSign from '../views/FaceSign.vue'        // 签到
import BehaviorMonitor from '../views/BehaviorMonitor.vue'  // 行为监测

const routes = [
  { path: '/', name: 'Index', component: Index },
  { path: '/face-sign', name: FaceSign, component: FaceSign },
  { path: '/behavior-monitor', name: BehaviorMonitor, component: BehaviorMonitor }
]

const router = createRouter({
  history: createWebHistory(),
  routes
})

export default router