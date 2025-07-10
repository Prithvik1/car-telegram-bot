import logging
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from car_ai_agent import (
    load_data, preprocess_data, train_model,
    map_needs_to_type, predict_selected_car
)
import pandas as pd
import google.generativeai as genai


import os
os.environ["GOOGLE_API_KEY"] = "AIzaSyCmfOTdnJQlRSHK_2tZaDvlfmIekc08T8c"
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


df = load_data()
X, y, encoder, cat_cols, num_cols = preprocess_data(df)
model, mae = train_model(X, y)


user_states = {}


def ask_gemini_to_estimate_text(car_details, ml_price):
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
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

Based on this, please respond with only your refined resale price in INR — **just the number**, no currency symbol, no explanation.
"""
    response = gemini_model.generate_content(prompt)
    return response.text.strip()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("👋 Hello! Type 'I want a car' to begin car selection.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip().lower()
    user_id = update.effective_user.id

    if text == "i want a car":
        user_states[user_id] = {"step": "need"}
        await update.message.reply_text("🚘 What type of car are you looking for? (e.g., family, sporty, budget, off-road, luxury):")
    elif user_id in user_states:
        state = user_states[user_id]
        step = state["step"]

        if step == "need":
            state["need"] = text
            state["step"] = "fuel"
            await update.message.reply_text("⛽ Preferred fuel type? (Petrol, Diesel, Hybrid, Electric):")
        elif step == "fuel":
            state["fuel"] = text.capitalize()
            state["step"] = "year"
            await update.message.reply_text("📅 Minimum year of manufacture you want (e.g., 2020):")
        elif step == "year":
            try:
                state["year"] = int(text)
                state["step"] = "done"

                
                regex = map_needs_to_type(state["need"])
                matches = df[
                    (df["Type"].str.contains(regex, case=False, na=False)) &
                    (df["Fuel Type"].str.lower() == state["fuel"].lower()) &
                    (df["Year"] >= state["year"])
                ]

                if matches.empty:
                    await update.message.reply_text("❌ No matching cars found.")
                    del user_states[user_id]
                    return

                state["matches"] = matches.reset_index()
                msg = "\n".join(
                    [f"{i}. {row['Brand']} {row['Model']} ({row['Year']}, {row['Type']}, {row['Fuel Type']})"
                     for i, row in state["matches"].iterrows()]
                )
                await update.message.reply_text(f"✅ Found {len(matches)} cars:\n\n{msg}\n\n🎯 Enter the number of the car you want to select:")
            except ValueError:
                await update.message.reply_text("❗ Please enter a valid year (e.g., 2020).")
        elif step == "done":
            try:
                index = int(text)
                matches = user_states[user_id]["matches"]
                if index < 0 or index >= len(matches):
                    raise IndexError
                selected_car = matches.loc[index]
                await update.message.reply_text("🔍 Estimating resale value...")

                
                input_data = selected_car.drop("Ex-Showroom-Price")
                input_df = pd.DataFrame([input_data])

                user_encoded = encoder.transform(input_df[cat_cols])
                user_encoded_df = pd.DataFrame(user_encoded, columns=encoder.get_feature_names_out(cat_cols))
                user_final_df = pd.concat([user_encoded_df, input_df[num_cols].reset_index(drop=True)], axis=1)

                ml_price = model.predict(user_final_df)[0]

                
                resale_price = ask_gemini_to_estimate_text(selected_car, ml_price)

                await update.message.reply_text(
                    f"💸 Based on your selected car, the estimated **resale value** is:\n\n👉 ₹{resale_price}"
                )
                del user_states[user_id]
            except (ValueError, IndexError):
                await update.message.reply_text("❌ Invalid selection. Please enter a valid number from the list.")
    else:
        await update.message.reply_text("💬 Type 'I want a car' to begin the process.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    app = ApplicationBuilder().token(os.getenv("TELEGRAM_BOT_TOKEN")).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    print("🤖 Bot is running...")
    app.run_polling()
