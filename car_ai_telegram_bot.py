import logging
import os
import pandas as pd
import google.generativeai as genai
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler, CallbackQueryHandler,
    ContextTypes, filters
)
from car_ai_agent import (
    load_data, preprocess_data, train_model,
    map_needs_to_type
)

# Load environment
from dotenv import load_dotenv
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
logging.basicConfig(level=logging.INFO)

# Load and train model
df = load_data()
X, y, encoder, cat_cols, num_cols = preprocess_data(df)
model, mae = train_model(X, y)
user_states = {}

# Gemini refinement
def ask_gemini_to_estimate_text(car_details, ml_price):
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are a car pricing expert.

Here is the car selected by the user:
- Brand: {car_details['Brand']}
- Model: {car_details['Model']}
- Year: {car_details['Year']}
- Type: {car_details['Type']}
- Fuel Type: {car_details['Fuel Type']}
- Original Ex-Showroom Price: ₹{car_details['Ex-Showroom-Price']:,}

A machine learning model has estimated the current resale value at ₹{int(ml_price):,}.

Based on this, please respond with only your refined resale price in INR — just the number, give it in the indian format, Lakhs, crores.
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


# /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Type 'I want a car' to begin.")


# Trigger phrase
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message.text.strip().lower() == "i want a car":
        user_id = update.effective_user.id
        user_states[user_id] = {"step": "need"}
        keyboard = [
            [InlineKeyboardButton("Family", callback_data="need:family")],
            [InlineKeyboardButton("Sporty", callback_data="need:sporty")],
            [InlineKeyboardButton("Budget", callback_data="need:budget")],
            [InlineKeyboardButton("Off-road", callback_data="need:offroad")],
            [InlineKeyboardButton("Luxury", callback_data="need:luxury")]
        ]
        await update.message.reply_text(
            "🚘 What type of car are you looking for?",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )
    else:
        await update.message.reply_text("💬 Please type 'I want a car' to begin.")


# Handle button selections
async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id

    if user_id not in user_states:
        user_states[user_id] = {}

    state = user_states[user_id]

    if query.data.startswith("need:"):
        state["need"] = query.data.split(":")[1]
        state["step"] = "fuel"
        keyboard = [
            [InlineKeyboardButton("Petrol", callback_data="fuel:Petrol")],
            [InlineKeyboardButton("Diesel", callback_data="fuel:Diesel")],
            [InlineKeyboardButton("Hybrid", callback_data="fuel:Hybrid")],
            [InlineKeyboardButton("Electric", callback_data="fuel:Electric")]
        ]
        await query.message.reply_text("⛽ Choose a fuel type:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("fuel:"):
        state["fuel"] = query.data.split(":")[1]
        state["step"] = "year"
        keyboard = [
            [InlineKeyboardButton("2018+", callback_data="year:2018")],
            [InlineKeyboardButton("2020+", callback_data="year:2020")],
            [InlineKeyboardButton("2022+", callback_data="year:2022")],
        ]
        await query.message.reply_text("📅 Choose minimum year of manufacture:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif query.data.startswith("year:"):
        year = int(query.data.split(":")[1])
        state["year"] = year
        state["step"] = "select_car"

        regex = map_needs_to_type(state["need"])
        matches = df[
            (df["Type"].str.contains(regex, case=False, na=False)) &
            (df["Fuel Type"].str.lower() == state["fuel"].lower()) &
            (df["Year"] >= year)
        ]

        if matches.empty:
            await query.message.reply_text("❌ No matching cars found.")
            del user_states[user_id]
            return

        state["matches"] = matches.reset_index()

        keyboard = []
        for i, row in state["matches"].iterrows():
            label = f"{i}. {row['Brand']} {row['Model']} ({row['Year']})"
            callback_data = f"car:{i}"
            keyboard.append([InlineKeyboardButton(label, callback_data=callback_data)])

        await query.message.reply_text(
            f"✅ Found {len(matches)} cars. Choose one:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif query.data.startswith("car:"):
        selected_index = int(query.data.split(":")[1])
        matches = user_states[user_id].get("matches")

        if matches is None or selected_index >= len(matches):
            await query.message.reply_text("❌ Invalid car selection.")
            return

        selected_car = matches.loc[selected_index]
        await query.message.reply_text("🔍 Estimating resale value...")

        input_data = selected_car.drop("Ex-Showroom-Price")
        input_df = pd.DataFrame([input_data])

        user_encoded = encoder.transform(input_df[cat_cols])
        user_encoded_df = pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(cat_cols))
        user_final_df = pd.concat([user_encoded_df, input_df[num_cols].reset_index(drop=True)], axis=1)

        ml_price = model.predict(user_final_df)[0]
        resale_price = ask_gemini_to_estimate_text(selected_car, ml_price)

        await query.message.reply_text(f"💸 Gemini estimates resale value: ₹{resale_price}")
        del user_states[user_id]


# Run the bot
if __name__ == "__main__":
    app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(CallbackQueryHandler(handle_callback))
    print("🤖 Bot is running...")
    app.run_polling()
