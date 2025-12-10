-- Neon Postgres Schema for Physical AI Book User Management

-- Users table (BetterAuth will manage most of this)
-- We'll extend it with our custom fields

-- User profiles table for storing additional user information
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL UNIQUE, -- This links to BetterAuth's user ID
    software_background TEXT, -- Possible values: 'beginner', 'intermediate', 'advanced', 'expert'
    hardware_background TEXT, -- Possible values: 'none', 'basic', 'intermediate', 'advanced'
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES better_auth_user(id) ON DELETE CASCADE
);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id TEXT NOT NULL,
    preferred_language TEXT DEFAULT 'en', -- en, ur, ru, ar, de
    personalization_enabled BOOLEAN DEFAULT true,
    dark_mode_enabled BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES better_auth_user(id) ON DELETE CASCADE
);

-- Ensure foreign key constraints work with BetterAuth
-- Note: The better_auth_user table will be created by BetterAuth itself

-- Indexes for better performance
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);
CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);
CREATE INDEX IF NOT EXISTS idx_user_profiles_software_bg ON user_profiles(software_background);
CREATE INDEX IF NOT EXISTS idx_user_profiles_hardware_bg ON user_profiles(hardware_background);

-- Function to update the updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers to automatically update the updated_at column
CREATE TRIGGER update_user_profiles_updated_at
    BEFORE UPDATE ON user_profiles
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Insert default preferences for existing users (if any)
-- This would typically be run after the schema is applied
-- INSERT INTO user_preferences (user_id, preferred_language, personalization_enabled)
-- SELECT id, 'en', true FROM better_auth_user
-- WHERE id NOT IN (SELECT user_id FROM user_preferences);