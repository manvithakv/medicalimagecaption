import mongoose from 'mongoose';
const userSchema = new mongoose.Schema({
    username:{
        type:String,
        required:true,
        unique: true
    },
    email:{
        type:String,
        required:true,
        unique:true
    },
    password:{
        type:String,
        required:true
    },
    phone:{
        type:Number,
        required:true
    }
})

const userModel= mongoose.model("User",userSchema)

export {userModel as User}