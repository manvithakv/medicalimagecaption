
import {BrowserRouter,Routes,Route} from 'react-router-dom'
import Signup from './components/Signup'
import Login from './components/login'
import Home from './components/Home'
import Otp from './components/otp'
import DLModelPage from './components/DLModelPage'
import LLMpage from './components/LLMpage'
import LLM from './components/LLM'
import LLMoriginal2 from './components/LLMoriginal2'

function App() {
  return(
    <BrowserRouter>
      <Routes>
        <Route path='/signup' element={<Signup/>}></Route>
        <Route path='/' element={<Login/>}></Route>
        <Route path='/home' element={<Home/>}></Route>
        <Route path='/otp' element={<Otp/>}></Route>
        <Route path='/dlmodel' element={<DLModelPage/>}></Route>
        <Route path='/llmmodel' element={<LLMpage/>}></Route>
        <Route path='/llmmodel2' element={<LLM/>}></Route>
        <Route path='/llmmodel3' element={<LLMoriginal2/>}></Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App;
