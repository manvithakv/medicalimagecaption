import React, { useState } from 'react'
import '../App.css'
import Axios from 'axios'
import {Link,useNavigate} from 'react-router-dom';

const Signup = () => {
    const [username, setUsername] = useState('')
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [phone,setPhone]=useState('')

    const navigate= useNavigate()
  
    const handelSubmit = (e) => {
        e.preventDefault()
        Axios.post('http://localhost:5000/auth/Signup', { username, email, password ,phone
        }).then(response => {
            if(response.data.status)
            {
                alert(response.data.message)
                navigate('/otp',{ state: { email } });
            }else{
            alert(response.data.message)}
        }).catch(err => {
            console.log(err)
        })
    }
    return (
        <div className='sign-up-container'>
            <h2>Sign up</h2>
            <form className='sign-up-form' onSubmit={handelSubmit}>
                <label htmlFor="username">Username:</label>
                <input type="text" placeholder="Username" onChange={(e) => setUsername(e.target.value)} />

                <label htmlFor="email">Email:</label>
                <input type="email" autoComplete='off' placeholder="Email" onChange={(e) => setEmail(e.target.value)} />

                <label htmlFor="password">password:</label>
                <input type="password" placeholder="********" onChange={(e) => setPassword(e.target.value)} />

                <label htmlFor="phone">Mobile Number:</label>
                <input type="phone" autoComplete='off' placeholder="phone number" onChange={(e) => setPhone(e.target.value)} />

                <button type='submit'>Signup</button>
                <p>already have an account? <Link to='/'>login</Link></p>
            </form>
        </div>
    )
}

export default Signup
