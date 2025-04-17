import React, { useState } from 'react'
import '../App.css'
import Axios from 'axios'
import { Link, useNavigate } from 'react-router-dom';

const Login = () => {
    const [email, setEmail] = useState('')
    const [password, setPassword] = useState('')
    const [showPopup, setShowPopup] = useState(false);

    const navigate = useNavigate()

    Axios.defaults.withCredentials = true;
 
    const handelSubmit = (e) => {
        e.preventDefault()
        Axios.post('http://localhost:5000/auth/', {
            email,
            password
        }).then(response => {
            if (response.data.status) {
                alert(response.data.message)
                navigate('/home')
            } else {
                setShowPopup(true); // Set state to show popup for user not registered
            }
        }).catch(err => {
            console.log(err)
        })
    }

    return (
        <div className='sign-up-container'>
            <h2>Login</h2>
            <form className='sign-up-form' onSubmit={handelSubmit}>
                <label htmlFor="email">Email:</label>
                <input type="email" autoComplete='off' placeholder="Email" onChange={(e) => setEmail(e.target.value)} />

                <label htmlFor="password">Password:</label>
                <input type="password" placeholder="********" onChange={(e) => setPassword(e.target.value)} />

                <button type='submit'>Login</button>
                <p>Don't have an account? <Link to='/Signup'>Signup</Link></p>
            </form>

            {/* Popup message */}
            {showPopup && (
                <div className="popup">
                    <p>User not registered</p>
                    <button onClick={() => setShowPopup(false)}>Close</button>
                </div>
            )}
        </div>
    )
}

export default Login
