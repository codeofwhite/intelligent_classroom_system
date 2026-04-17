import { createApp } from 'vue'
import App from './App.vue'
import router from './router/index.js' // 引入路由

const app = createApp(App)
app.use(router) // 启用路由
app.mount('#app')