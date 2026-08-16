import { loadEnv, defineConfig } from '@medusajs/framework/utils'

loadEnv(process.env.NODE_ENV || 'development', process.cwd())

module.exports = defineConfig({
  projectConfig: {
    databaseUrl: process.env.DATABASE_URL,
    // The create-medusa-app scaffold generates a .env with REDIS_URL but
    // does not wire it into projectConfig by default -- without this line
    // Medusa silently falls back to an in-memory event bus/workflow engine
    // ("redisUrl not found. A fake redis instance will be used."), which
    // defeats the point of running a real Redis container.
    redisUrl: process.env.REDIS_URL,
    http: {
      storeCors: process.env.STORE_CORS!,
      adminCors: process.env.ADMIN_CORS!,
      authCors: process.env.AUTH_CORS!,
      jwtSecret: process.env.JWT_SECRET,
      cookieSecret: process.env.COOKIE_SECRET,
    }
  }
})
