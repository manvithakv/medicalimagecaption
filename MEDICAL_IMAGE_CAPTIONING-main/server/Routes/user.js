import express from 'express';
import bcryt from'bcrypt';
import jwt from 'jsonwebtoken';
import nodemailer from 'nodemailer';
import twilio from 'twilio';
import dotenv from 'dotenv'

dotenv.config()

import { User } from '../models/User.js';
import OTP from '../models/userOtp.js';
const router = express.Router();

const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user:process.env.EMAIL,
        pass: process.env.PASSWORD
    }
});

const accountSid=process.env.TWILIO_SID;
const authToken=process.env.TWILIO_AUTH_TOKEN;
const client=twilio(accountSid,authToken);

const generateOTP = () => {
    return Math.floor(100000 + Math.random() * 900000).toString();
};


// Send OTP via email and store it in the database
router.post('/sendOTP', async (req, res) => {
    const { email} = req.body;
    const user=await User.findOne({email})
    const otp = generateOTP();

    // Save OTP to the database
    const newOTP = new OTP({
        email,
        otp
    });

    try {
        await newOTP.save();
    } catch (error) {
        console.log(error);
        return res.status(500).send({ status: false, message: 'Failed to save OTP' });
    }

    const mailOptions = {
        from:process.env.EMAIL,
        to: email,
        subject: 'OTP Verification',
        text: `Your OTP for verification is: ${otp}`
    };

    transporter.sendMail(mailOptions, (error, info) => {
        if (error) {
            console.log(error);
            res.status(500).send({ status: false, message: 'Failed to send OTP' });
        } else {
            console.log('Email sent: ' + info.response);
            res.status(200).send({ status: true, message: 'OTP sent successfully' });
        }
    })
    /*await client.messages.create({
        body: `YOUR OTP IS: ${otp}`,
        to: `+91${user.phone}`,
        from: '+12058391344',
    });*/
});

// Verify OTP
router.post('/verifyOTP', async (req, res) => {
    const { email, otp } = req.body;

    // Retrieve OTP from the database based on the email
    const otpRecord = await OTP.findOne({ email });

    if (!otpRecord) {
        return res.json({ status: false, message: 'OTP not found' });
    }

    // Compare the OTP received with the one stored in the database
    if (otp === otpRecord.otp) {
        res.json({ status: true, message: 'OTP verified successfully' });
    } else {
        res.json({ status: false, message: 'Incorrect OTP' });
    }
});

router.post('/Signup', async (req,res)=>{
    const {username,email,password,phone}=req.body;
    const user= await User.findOne({email})
    if(user)
    {
        console.log("user exist")
        return res.json({ status:false, message:"user already exist"})
    }
    const hashpassword = await bcryt.hash(password,10)
    const newUser =new User({
        username,
        email,
        password: hashpassword,
        phone,
    })
    await newUser.save()
    return res.json({ status:true, message:"record registered"})
})

router.post('/',async (req,res)=>{
    const {email,password}= req.body;
    const user = await User.findOne({email})
    if(!user)
    {
        return res.json({message:"user is not registered"})
    }

    const validPassword = await bcryt.compare(password,user.password)
    if(!validPassword)
    {
        return res.json({message:"incorrect password"})
    }

    const token = jwt.sign({username:user.username},process.env.KEY,{expiresIn:'1h'})
    res.cookie('token',token,{httpOnly: true,maxAge:360000})
    return res.json({status:true, message:"login succesfull"})
})

export {router as UserRouter};