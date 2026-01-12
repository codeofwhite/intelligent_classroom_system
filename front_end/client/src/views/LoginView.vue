<template>
  <div class="login-container">
    <div class="login-box">
      <h2>🔑 欢迎回来</h2>
      <p>请选择身份进入智能课堂</p>
      <div class="role-selector">
        <button @click="emitLogin('student')">我是学生</button>
        <button @click="emitLogin('parent')">我是家长</button>
      </div>
    </div>
  </div>
</template>

<script setup>
// 定义可以发出的事件
const emit = defineEmits(['login-success'])

const emitLogin = (role) => {
  // 告诉父组件 App.vue，“有人登录了，角色是XXX”
  emit('login-success', role)
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 80vh;
}
.login-box {
  background: white;
  padding: 40px;
  border-radius: 20px;
  text-align: center;
  box-shadow: 0 10px 25px rgba(0,0,0,0.1);
}
.role-selector {
  display: flex;
  gap: 20px;
  margin-top: 30px;
}
button {
  padding: 12px 24px;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  background: #4a90e2;
  color: white;
}
</style>