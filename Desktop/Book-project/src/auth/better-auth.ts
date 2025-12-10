import { betterAuth } from "better-auth";
import { postgresAdapter } from "better-auth/adapters/postgres";

// Initialize BetterAuth with our configuration
export const auth = betterAuth({
  database: postgresAdapter({
    url: process.env.DATABASE_URL!, // Requires DATABASE_URL in environment
    provider: "postgres"
  }),
  socialProviders: {
    // Add social login providers as needed
    // google: {
    //   clientId: process.env.GOOGLE_CLIENT_ID!,
    //   clientSecret: process.env.GOOGLE_CLIENT_SECRET!,
    // },
    // github: {
    //   clientId: process.env.GITHUB_CLIENT_ID!,
    //   clientSecret: process.env.GITHUB_CLIENT_SECRET!,
    // },
  },
  // Custom user fields for software and hardware background
  user: {
    additionalFields: {
      softwareBackground: {
        type: "string",
        required: false, // Not required during initial signup, but collected later
        private: false, // This field can be accessed publicly if needed
      },
      hardwareBackground: {
        type: "string",
        required: false, // Not required during initial signup, but collected later
        private: false, // This field can be accessed publicly if needed
      }
    }
  },
  // Session configuration
  session: {
    expiresIn: 7 * 24 * 60 * 60, // 7 days
    rememberMe: true, // Allow "Remember me" functionality
    updateAge: 24 * 60 * 60, // Update session every 24 hours
  },
  // Email & password configuration
  emailAndPassword: {
    enabled: true,
    requireEmailVerification: false, // Set to true in production
    password: {
      enabled: true,
      minLength: 8,
    }
  },
  // Account configuration
  account: {
    accountLinking: {
      enabled: true, // Allow linking multiple auth methods to one account
    }
  },
  // Cookie configuration for security
  cookies: {
    domain: process.env.NODE_ENV === 'production' ? '.your-book-site.com' : undefined,
    path: '/',
    sameSite: 'lax', // CSRF protection
    secure: process.env.NODE_ENV === 'production', // Only over HTTPS in production
  },
  // Rate limiting to prevent abuse
  rateLimit: {
    window: 60 * 10, // 10 minute window
    max: 100, // Max 100 requests per window
  },
  // Localization (if needed)
  email: {
    // Custom email templates can be added here
    // For account verification, password reset, etc.
  }
});

// Export types for TypeScript
export type User = typeof auth.$Infer.Session.user;