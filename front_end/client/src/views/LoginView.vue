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
const emit = defineEmits(['login-success'])
const emitLogin = (role) => {
  emit('login-success', role)
}
</script>

<style scoped>
.login-container {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 100vh;
  background: linear-gradient(to bottom right, #eef5ff, #d4e7ff);
}
.login-box {
  background: white;
  padding: 50px 40px;
  border-radius: 24px;
  text-align: center;
  box-shadow: 0 15px 35px rgba(0, 0, 0, 0.08);
  width: 340px;
}
.login-box h2 {
  margin: 0 0 8px;
  font-size: 22px;
}
.login-box p {
  margin: 0 0 30px;
  color: #666;
  font-size: 14px;
}
.role-selector {
  display: flex;
  gap: 16px;
  justify-content: center;
}
button {
  padding: 14px 28px;
  border: none;
  border-radius: 12px;
  cursor: pointer;
  background: #4a90e2;
  color: white;
  font-size: 15px;
  transition: 0.2s;
  flex: 1;
}
button:hover {
  background: #357abd;
  transform: translateY(-2px);
}
</style>