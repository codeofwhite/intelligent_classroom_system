import { createApp } from 'vue'
import App from './App.vue'
import router from './router' // 引入刚才创建的配置

const app = createApp(App)
app.use(router) // 使用插件
app.mount('#app')
