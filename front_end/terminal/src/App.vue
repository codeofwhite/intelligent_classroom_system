<template>
  <div id="app">
    <nav>
      <router-link to="/">控制台</router-link>
      <router-link to="/face-sign">人脸识别签到</router-link>
      <router-link to="/behavior-monitor">学生行为监测</router-link>
    </nav>
    <router-view />
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}
#app {
  min-height: 100vh;
  background: #000;
  color: #0f0;
  font-family: "Courier New", monospace;
}
nav {
  padding: 1rem;
  background: #111;
  display: flex;
  gap: 20px;
}
nav a {
  color: #0f0;
  text-decoration: none;
  font-size: 18px;
}
</style>